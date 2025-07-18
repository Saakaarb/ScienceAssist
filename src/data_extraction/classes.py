import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean
from unstructured.documents.elements import NarrativeText,Text, PageBreak, Element
from unstructured.chunking.title import chunk_by_title
from datasets import Dataset
from datasets import load_from_disk
import shutil
import signal
import time
from src.utils.config_loader import get_config_value
# This class is used to extract the data from the downloaded data directory.
# it extracts chunks from PDFs and combines them with metadata to create a dataset.
# the dataset is saved in the processed_data directory.
# the dataset is saved in a 
# the csv file contains the following columns:

class DataExtractor:
    def __init__(self, processed_data_location: Path, raw_dataset_location: Path, max_characters: int, new_after_n_chars: int, config: dict) -> None:
        self.processed_data_location=processed_data_location
        self.raw_dataset_location=raw_dataset_location
        self.max_characters=max_characters
        self.new_after_n_chars=new_after_n_chars
        self.config=config
        self.min_characters=get_config_value(self.config, 'text_processing.min_characters', 100)
        self.timeout_seconds=get_config_value(self.config, 'text_processing.timeout_seconds', 180)
        
    def fetch_raw_dataset(self) -> tuple[pd.DataFrame, list[str]]:
        """
        Load and preprocess the raw dataset metadata from the data download pipeline.
        
        This function reads the metadata CSV file created by the data download pipeline,
        filters out papers that failed to download (where Filename is NaN), and extracts
        the list of successfully downloaded PDF filenames for processing.
        
        Args:
            None (uses self.raw_dataset_location from class instance)
            
        Returns:
            tuple[pd.DataFrame, list[str]]: A tuple containing:
                - raw_metadata_df (pd.DataFrame): Cleaned metadata dataframe with failed downloads removed
                - pdf_file_names (list[str]): List of PDF filenames to process
                
        Process:
        1. Construct path to metadata.csv file in raw dataset location
        2. Load metadata CSV file into pandas DataFrame
        3. Filter out rows where Filename is NaN (failed downloads)
        4. Extract list of PDF filenames from the cleaned DataFrame
        5. Return both the cleaned DataFrame and filename list
        
        Note:
            The metadata CSV contains columns: Index, Title, Abstract, DOI, ArXiv Link, 
            Authors, Published Date, and Filename. Only papers with valid filenames
            are included in the returned data.
        """

        raw_metadata_filepath=self.raw_dataset_location/Path("metadata.csv")

        raw_metadata_df=pd.read_csv(raw_metadata_filepath)

        # Delete rows where Filename is NaN (failed downloads)
        
        raw_metadata_df = raw_metadata_df[~raw_metadata_df['Filename'].isna()]

        # get the pdf file names
        pdf_file_names=raw_metadata_df['Filename'].tolist()

        return raw_metadata_df,pdf_file_names

    def clean_text(self,text: str) -> str:
        # Remove LaTeX-style math and equation environments
        text = re.sub(r"\$.*?\$", "", text)
        text = re.sub(r"\\begin\{.*?\}.*?\\end\{.*?\}", "", text, flags=re.DOTALL)

        # Remove anything inside square brackets [] and round brackets ()
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\(.*?\)", "", text)

        # Remove figure/table captions
        text = re.sub(r"(Figure|Table)\s*\d+[:\.]", "", text, flags=re.IGNORECASE)

        # Remove References or Bibliography section
        text = re.sub(r"(?i)(references|bibliography)\s*[\n\r].*", "", text, flags=re.DOTALL)

        # Remove "cid:" artifacts like "cid:123"
        text = re.sub(r"cid:\d+", "", text)

        # Remove dash `-` if surrounded by non-alphanumeric or spaces
        text = re.sub(r"(?<=[^a-zA-Z0-9\s])-(?=[^a-zA-Z0-9\s])|(?<=\s)-(?=[^a-zA-Z0-9])|(?<=[^a-zA-Z0-9])-(?=\s)|(?<=\s)-(?=\s)", "", text)

        # Remove all non-ASCII characters
        text = text.encode("ascii", errors="ignore").decode()

        # Collapse excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Convert to lowercase
        text = text.lower()

        return text.strip()


    def extract_elements_from_pdf(self,pdf_file_path: Path) -> list[Element]:
        elements = partition_pdf(filename=pdf_file_path,
                                 strategy="fast",
                                 extract_images_in_pdf=False,
                                 )
        return elements

    def timeout_handler(self, signum, frame):
        """Handler for timeout signal"""
        raise TimeoutError("PDF processing timed out after 180 seconds")

    def extract_elements_from_pdf_with_timeout(self, pdf_file_path: Path, timeout_seconds: int = 180) -> list[Element]:
        """
        Extract elements from PDF with timeout protection.
        
        Args:
            pdf_file_path: Path to the PDF file
            timeout_seconds: Maximum time to wait for processing (default: 180)
            
        Returns:
            elements: Extracted elements if successful
            
        Raises:
            TimeoutError: If processing takes longer than timeout_seconds
        """
        # Set up timeout signal
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            elements = self.extract_elements_from_pdf(pdf_file_path)
            signal.alarm(0)  # Cancel the alarm
            return elements
        except TimeoutError:
            signal.alarm(0)  # Cancel the alarm
            raise
        except Exception as e:
            signal.alarm(0)  # Cancel the alarm
            raise e

    def get_curr_pdf_metadata(self,pdf_file_name: str,raw_metadata_df: pd.DataFrame) -> dict:
        """
        Extract and preprocess metadata for a specific PDF file from the raw dataset.
        
        This function retrieves the metadata row corresponding to a given PDF filename,
        removes the Abstract field (which is not needed for the processed dataset),
        and converts all values to strings to ensure compatibility with the dataset format.
        
        Args:
            pdf_file_name (str): Name of the PDF file to get metadata for
            raw_metadata_df (pd.DataFrame): Raw metadata dataframe from fetch_raw_dataset()
            
        Returns:
            dict: Dictionary containing cleaned metadata for the specified PDF file
            
        Process:
        1. Find the row in raw_metadata_df where Filename matches pdf_file_name
        2. Convert the row to a dictionary
        3. Remove the 'Abstract' field from the metadata
        4. Convert all values to strings, replacing NaN values with "N/A"
        5. Return the cleaned metadata dictionary
        
        Note:
            The returned dictionary contains all metadata fields except 'Abstract':
            Index, Title, DOI, ArXiv Link, Authors, Published Date, and Filename.
            All values are converted to strings to prevent PyArrow type conversion issues
            when creating the final dataset.
        """
        
        # get the row where the filename matches
        curr_pdf_metadata=raw_metadata_df[raw_metadata_df['Filename']==pdf_file_name].iloc[0].to_dict()
        
        # Remove Abstract field
        curr_pdf_metadata.pop('Abstract')
        
        # Convert all values to strings to prevent PyArrow type conversion issues
        for key, value in curr_pdf_metadata.items():
            if pd.isna(value):
                curr_pdf_metadata[key] = "N/A"
            else:
                curr_pdf_metadata[key] = str(value)

        return curr_pdf_metadata

    # extract data into chunks and store in a Dataset object
    def extract_and_save_data(self) -> None:
        """
        Main pipeline function to extract text chunks from PDFs and create a processed dataset.
        
        This function orchestrates the complete data extraction workflow: loading raw metadata,
        processing each PDF file to extract text elements, chunking the text into manageable
        pieces, cleaning the text, and combining it with metadata to create a HuggingFace
        Dataset that is saved to disk.
        
        Args:
            None (uses class instance variables for configuration)
            
        Returns:
            None (saves dataset to self.processed_data_location)
            
        Workflow:
        1. Load raw dataset metadata and PDF filenames
        2. Iterate through each PDF file with progress tracking
        3. Extract PDF elements with timeout protection (skip slow files)
        4. Chunk extracted elements by title with character limits
        5. Clean each text chunk using regex-based text processing
        6. Filter chunks below minimum character threshold
        7. Combine cleaned text with paper metadata and page numbers
        8. Create HuggingFace Dataset from all chunk data
        9. Save dataset to disk in processed data location
        
        Error Handling:
        - Timeout protection for slow PDF processing (skips files taking > timeout_seconds)
        - Exception handling for individual PDF processing errors
        - Continues processing remaining files if one fails
        
        Files Created:
        - HuggingFace Dataset saved to self.processed_data_location containing:
          - text: Cleaned text chunks
          - metadata: Paper information (title, authors, DOI, etc.)
          - page_nos: Set of page numbers for each chunk
        """

        raw_metadata_df,pdf_file_names=self.fetch_raw_dataset()
        all_chunk_data=[]

        for pdf_file_name in tqdm(pdf_file_names):
            print("Processing document: ",pdf_file_name)
            curr_pdf_metadata=self.get_curr_pdf_metadata(pdf_file_name,raw_metadata_df)
            pdf_file_path=self.raw_dataset_location/Path(pdf_file_name)
            
            try:
                # Extract elements with timeout protection
                elements = self.extract_elements_from_pdf_with_timeout(pdf_file_path, timeout_seconds=self.timeout_seconds)
            except TimeoutError:
                print(f"Timeout: Skipping {pdf_file_name} - processing took more than {self.timeout_seconds} seconds")
                continue
            except Exception as e:
                print(f"Error processing {pdf_file_name}: {e}")
                continue
        
            #chunk and clean data
            #---------------------------------------------------------
            # convert PDF into text
            #TODO parallelize across CPUs
            # chunk elements
            
            chunks = chunk_by_title(elements,max_characters=self.max_characters,new_after_n_chars=self.new_after_n_chars)

            # for every chunk, clean and store text along with extra metadata  

            #print("Cleaning and storing extracted chunks...")
            # clean text of each chunk and assemble dataset
            for chunk in chunks:
                chunk_data={} # reinit chunk data
                # regex based cleaning
                clean_chunk=self.clean_text(chunk.text)

                if len(clean_chunk)<self.min_characters:
                    continue

                page_nos = [e.metadata.page_number for e in chunk.metadata.orig_elements]

                chunk_data['text']=clean_chunk
                chunk_data.update(curr_pdf_metadata)
                chunk_data.update({'page_nos':set(page_nos)}) # set for unique number storage

                all_chunk_data.append(chunk_data.copy())

        print("Sample Chunks:")
        print(all_chunk_data[0])
        print(all_chunk_data[1])
        print(all_chunk_data[2])
        print(all_chunk_data[3])
        print(all_chunk_data[4])
        print(all_chunk_data[5])
        print(all_chunk_data[6])
        print("Total number of chunks extracted: ",len(all_chunk_data))
        dataset = Dataset.from_list(all_chunk_data)
        dataset.save_to_disk(self.processed_data_location)