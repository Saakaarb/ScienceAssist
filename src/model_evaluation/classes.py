from pathlib import Path
from src.utils.config_loader import ConfigLoader,get_config_value,load_pipeline_config
from src.model_inference.classes import ModelInference
import os
import datetime
from bertopic import BERTopic
from datasets import load_from_disk, Dataset
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.lib.LLM.classes import LLMBase
from src.lib.LLM.helper_functions import create_LLM_instance
from tqdm import tqdm
import random

# This class is used to evaluate the performance of the model.
# Steps:
# 1. generate eval data from dataset using GPT in (query,answer) format for a given dataset (done ONCE per dataset)

        #i. Bertopic to create N topics from the given context
        #ii. Get: popular topic keywords, and the relevant identified chunks
        #iii. generate N questions from the popular topic keywords and the relevant identified chunks, the questions dataset will contain which chunks were used
        #iv. Pass the questions to the RAG model
        #v. Compare RAG model's answer with 
# 2.  Pass the queries to the model
# 3. Evaluate the model's performance


class ModelEvaluation:
    def __init__(self, exp_name, processed_dataset_name, model_name,config:dict):
        self.dataset_path=Path("data")/Path(exp_name)/Path("processed_data")/Path(processed_dataset_name)
        self.model_path=Path("models")/Path(exp_name)/Path(model_name)

        
        self.eval_dataset_path=Path("evaluations")/Path(exp_name)/Path(processed_dataset_name)/Path("eval_dataset")
        self.eval_results_path=Path("evaluations")/Path(exp_name)/Path(processed_dataset_name)/Path("eval_results")
        if not os.path.isdir(self.eval_dataset_path.parent.parent):
            os.makedirs(self.eval_dataset_path.parent.parent)
        if not os.path.isdir(self.eval_dataset_path.parent):
            os.makedirs(self.eval_dataset_path.parent)
        

        #dataset_config_path=dataset_path/Path("data_extraction_config.yaml")
        #model_config_path=model_path/Path("model_creation_config.yaml")

        # load config files that were saved with dataset and model
        self.dataset_config_loader=ConfigLoader(config_dir=self.dataset_path)
        self.model_config_loader=ConfigLoader(config_dir=self.model_path)

        self.dataset_config=self.dataset_config_loader.load_config("data_extraction")
        self.model_config=self.model_config_loader.load_config("model_creation")
        self.inference_config=load_pipeline_config("model_inference")

        self.eval_config=config
        self.llm_API_key=get_config_value(self.eval_config, "api.API_key_string")
        self.llm_model_name=get_config_value(self.eval_config, "model.LLM_model_name")
        self.llm_vendor_name=get_config_value(self.eval_config, "model.LLM_vendor_name")
        
        
        self.question_generation_instructions_file_path=Path("src/lib/LLM/LLM_instr_files/question_generation_instructions.txt")
        self.answer_generation_instructions_file_path=Path("src/lib/LLM/LLM_instr_files/answer_generation_instructions.txt")
        self.judge_instructions_file_path=Path("src/lib/LLM/LLM_instr_files/judge_instructions.txt")
        # some settings
        self.num_top_topics_for_questions=get_config_value(self.eval_config, "dataset_generation.num_top_topics_for_questions")
        self.num_random_topics_for_questions=get_config_value(self.eval_config, "dataset_generation.num_random_topics_for_questions")
        self.num_questions_per_topic=get_config_value(self.eval_config, "dataset_generation.num_questions_per_topic")
        self.topic_confidence_threshold=get_config_value(self.eval_config, "dataset_generation.topic_confidence_threshold")
        # datasets to generate
        self.chunks_for_openai_limits=490

    def clean_text_for_topic_analysis(self, text: str) -> str:
        """
        Clean text by removing English stop words and common fill words for better topic analysis.
        
        This function removes common English stop words, articles, prepositions, and other
        fill words that don't contribute meaningful content for topic modeling. It also
        performs basic text cleaning like removing extra whitespace and converting to lowercase.
        
        Args:
            text (str): Input text to be cleaned
            
        Returns:
            str: Cleaned text with stop words removed
            
        Note:
            This function downloads NLTK stop words if not already available.
        """
        try:
            # Download NLTK stop words if not already available
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {e}")
            # Fallback to basic stop words list
            stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs'}
        else:
            # Get NLTK stop words
            stop_words = set(stopwords.words('english'))
            
            # Add common scientific/academic fill words
            scientific_stop_words = {
                'paper', 'study', 'research', 'analysis', 'method', 'approach', 'technique',
                'result', 'conclusion', 'finding', 'data', 'dataset', 'experiment',
                'model', 'algorithm', 'system', 'framework', 'methodology', 'evaluation',
                'performance', 'comparison', 'investigation', 'examination', 'assessment',
                'review', 'survey', 'overview', 'summary', 'discussion', 'implication',
                'application', 'implementation', 'development', 'design', 'construction',
                'proposed', 'presented', 'described', 'demonstrated', 'showed', 'indicated',
                'suggested', 'revealed', 'observed', 'found', 'obtained', 'achieved',
                'improved', 'enhanced', 'optimized', 'evaluated', 'tested', 'validated',
                'verified', 'confirmed', 'established', 'determined', 'identified',
                'characterized', 'analyzed', 'examined', 'investigated', 'explored',
                'studied', 'researched', 'developed', 'designed', 'implemented',
                'proposed', 'presented', 'introduced', 'described', 'discussed',
                'considered', 'addressed', 'focused', 'concentrated', 'emphasized',
                'highlighted', 'noted', 'mentioned', 'reported', 'published',
                'available', 'accessible', 'obtainable', 'achievable', 'feasible',
                'practical', 'useful', 'effective', 'efficient', 'successful',
                'significant', 'important', 'relevant', 'appropriate', 'suitable',
                'adequate', 'sufficient', 'necessary', 'required', 'essential',
                'critical', 'crucial', 'vital', 'fundamental', 'basic', 'primary',
                'main', 'major', 'key', 'principal', 'central', 'core', 'essential'
            }
            stop_words.update(scientific_stop_words)
        
        # Convert to lowercase and tokenize
        text_lower = text.lower()
        tokens = word_tokenize(text_lower)
        
        # Remove stop words and short tokens (likely noise)
        cleaned_tokens = [
            token for token in tokens 
            if token not in stop_words 
            and len(token) > 2  # Remove very short tokens
            and not token.isnumeric()  # Remove pure numbers
            and not re.match(r'^[^\w\s]+$', token)  # Remove pure punctuation
        ]
        
        # Rejoin tokens with spaces
        cleaned_text = ' '.join(cleaned_tokens)
        
        # Remove extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text

    def load_dataset(self)->None:

        """
        Load the processed dataset from disk.
        
        This function loads the HuggingFace Dataset that was created by the data extraction
        pipeline, which contains the processed text chunks and metadata needed for
        retrieval-augmented generation.
        
        Args:
            None (uses self.path_to_dataset from class instance)
            
        Returns:
            None (sets self.dataset attribute)
        """

        if os.path.isdir(self.dataset_path):
            self.dataset = load_from_disk(self.dataset_path)
        else:
            raise ValueError(f"Dataset not found at {self.dataset_path}")

    def generate_question_for_single_topic(self, llm_instance, topic_keywords, chunks)->None:


        query=f"Generate {self.num_questions_per_topic} questions for the following topic using keywords:"
        for tuple_value in topic_keywords:
            query+=f"{tuple_value[0]}: \n"
        query+="\n"
        
        #query+="The following chunks are relevant to the topic: \n"
        #for chunk in chunks:
        #    query+=f"{chunk}\n"
        #    query+="\n"

        answer=llm_instance.query_model(query, self.question_generation_instructions_file_path)
        
        # split answer into questions
        all_questions=answer.split("---")

        #assert len(all_questions)==self.num_questions_per_topic, "Number of questions generated is not equal to the number of questions per topic"


        return all_questions

    def generate_questions_from_topics(self, topics_for_questions, chunks_for_topics)->None:
        

        reference_file_paths=[]
        print("Creating LLM instance...")
        llm_instance=create_LLM_instance(self.llm_API_key, reference_file_paths, self.llm_model_name, self.llm_vendor_name)

        all_questions=[]
        for i, topic in tqdm(enumerate(topics_for_questions),total=len(topics_for_questions),desc="Generating questions from topics")   :
            curr_topic_keywords=topics_for_questions[i]
            curr_chunks=chunks_for_topics[i]

            questions=self.generate_question_for_single_topic(llm_instance, curr_topic_keywords, curr_chunks)

            all_questions+=questions 

        return all_questions

    def change_chunk_format(self, all_topic_chunks: list[list[str]]) -> list[str]:
        """
        Change the format of the chunks to be used for the LLM vector database.
        """
        modified_chunks=[]
        
        for topic_chunks in all_topic_chunks:
            
            for chunk in topic_chunks:
                
                modified_chunks.append(chunk)

        return modified_chunks

    def generate_eval_data(self)->None:

        if os.path.isdir(self.eval_dataset_path):
            raise ValueError(f"Evaluation dataset already exists at {self.eval_dataset_path} and cannot be overwritten. Please delete it and run the evaluation pipeline again.")

        print("Generating evaluation data for given dataset")
        # loop to extract texts and clean them for topic analysis
        all_text=[]
        all_cleaned_text=[]

        for i,text_w_metadata in enumerate(self.dataset):

            text=text_w_metadata["text"]
            cleaned_text=self.clean_text_for_topic_analysis(text)

            all_text.append(text)  # Keep original text for reference
            all_cleaned_text.append(cleaned_text)  # Use cleaned text for topic analysis

            


        print("Running BERTopic, this can take a while...")
        topic_model=BERTopic()

        topics, probs = topic_model.fit_transform(all_cleaned_text)

        topic_info=topic_model.get_topic_info()
        doc_info=topic_model.get_document_info(all_text)
        
        # Get number of topics (excluding outlier topic -1)
        num_topics = len(topic_info[topic_info['Topic'] != -1])
        print(f"Number of topics found: {num_topics}")
    
        # accumulate topic info
        # first 10 topics are used, and a few random ones from the rest
        topics_for_questions=[]
        chunks_for_topics=[]
        topics_added = 0
        
        # top N topics
        for i in range(self.num_top_topics_for_questions):
            # Get all documents that are identified as being related to topic i using doc_info with confidence threshold
            topic_docs_mask = (doc_info['Topic'] == i) & (doc_info['Probability'] > self.topic_confidence_threshold)
            topic_docs_indices = doc_info[topic_docs_mask].index.tolist()
            topic_docs = [all_text[idx] for idx in topic_docs_indices]
            
            # Only add topic if it has high-confidence documents
            if len(topic_docs) > 0:
                topics_for_questions.append(topic_model.get_topic(i))
                chunks_for_topics.append(topic_docs)
                topics_added += 1
                
                print(f"Topic {i}: {len(topic_docs)} documents (confidence > {self.topic_confidence_threshold})")
            else:
                print(f"Topic {i}: Skipped (no documents with confidence > {self.topic_confidence_threshold})")
            
        '''
        # some random topics   
        import random
        
        # Get all available topics (excluding -1 and top topics already processed)
        available_topics = [t for t in topic_info['Topic'].unique() 
                           if t != -1 and t >= self.num_top_topics_for_questions]
        
        # Select random topics
        if len(available_topics) > 0:
            num_random_to_select = min(self.num_random_topics_for_questions, len(available_topics))
            random_topics = random.sample(available_topics, num_random_to_select)
            
            print(f"\nProcessing {num_random_to_select} random topics: {random_topics}")
            
            for i in random_topics:
                # Get all documents that are identified as being related to topic i using doc_info
                topic_docs_mask = (doc_info['Topic'] == i) & (doc_info['Probability'] > 0.5)
                topic_docs_indices = doc_info[topic_docs_mask].index.tolist()
                topic_docs = [all_text[idx] for idx in topic_docs_indices]
                
                # Only add topic if it has high-confidence documents
                if len(topic_docs) > 0:
                    topics_for_questions.append(topic_model.get_topic(i))
                    chunks_for_topics.append(topic_docs)
                    topics_added += 1
                    
                    print(f"Random Topic {i}: {len(topic_docs)} documents (confidence > 0.5)")
                else:
                    print(f"Random Topic {i}: Skipped (no documents with confidence > 0.5)")
        else:
            print(f"\nNo additional topics available for random selection")
        
        print(f"\nTotal topics selected for questions: {topics_added}")
        '''

        all_questions=self.generate_questions_from_topics(topics_for_questions, chunks_for_topics)

        llm_instance=create_LLM_instance(self.llm_API_key, [], self.llm_model_name, self.llm_vendor_name)

        # fetch processed database
        
        processed_chunks=self.change_chunk_format(chunks_for_topics)
        if len(processed_chunks)>self.chunks_for_openai_limits:
            #select random chunks from processed_chunks
            processed_chunks=random.sample(processed_chunks, self.chunks_for_openai_limits)
        print("Creating vector database...")
        llm_instance.create_vector_database(processed_chunks)

        # retrieve relevant chunks from OpenAI vector store for each question
        chunks_for_all_questions=[]
        # retrieving relevant chunks from remote vector database
        print("Retrieving relevant chunks from remote vector database")
        for question in tqdm(all_questions):
            chunks_for_question=[]
            vector_store_result=list(llm_instance.query_vector_store(question))
            
            # Extract text from VectorStoreSearchResponse objects
            for result in vector_store_result:
                # Each result is a VectorStoreSearchResponse object
                if hasattr(result, 'content') and result.content:
                    for content_item in result.content:
                        # Each content_item is a Content object with a 'text' attribute
                        if hasattr(content_item, 'text'):
                            chunks_for_question.append(content_item.text)

            chunks_for_all_questions.append(chunks_for_question)

        
        # save in Q/A format

        self.save_questions_answers_to_file(all_questions,chunks_for_all_questions,llm_instance)

    def save_questions_answers_to_file(self, all_questions: list[str], chunks_for_all_questions: list[list[str]], llm_instance: LLMBase) -> None:
        """
        
        """
        
        dataset=[]
        for i, question_string in tqdm(enumerate(all_questions),total=len(all_questions),desc="Generating answers for questions"):
            current_dataset_entry={}
            current_question=all_questions[i]
            current_dataset_entry["question"]=current_question
            
            current_chunks=chunks_for_all_questions[i]
            current_dataset_entry["context"]=current_chunks

            answer=llm_instance.query_model(current_question, self.answer_generation_instructions_file_path)
            current_dataset_entry["answer"]=answer

            dataset.append(current_dataset_entry)
        
        llm_instance.delete_vector_store()

        hf_dataset=Dataset.from_list(dataset)
        hf_dataset.save_to_disk(self.eval_dataset_path)
        
        print(f"✅ Saved combined query with {len(all_questions)} questions to: {self.eval_dataset_path}")

        return dataset


    def query_created_model(self, dataset: Dataset) -> None:
        """
        Query the created RAG model using the ModelInference class.
        
        This function initializes the ModelInference class with the trained model
        and uses it to generate answers for the provided questions.
        
        Args:
            questions (list[str]): List of questions to answer
            
        Returns:
            list[str]: List of answers generated by the RAG model
        """
        
        # Construct paths for model inference
        model_output_folder = self.model_path
        path_to_dataset = self.dataset_path
        
        

        LLM_vendor_name = get_config_value(self.inference_config, 'llm.LLM_vendor_name', 'openai')
        LLM_model_name = get_config_value(self.inference_config, 'llm.LLM_model_name', 'gpt-4.1')
        API_key_string = get_config_value(self.inference_config, 'api.API_key_string', 'OPENAI_API_KEY')
        LLM_instr_filename = Path(get_config_value(self.inference_config, 'paths.LLM_instr_filename', 'src/lib/LLM/LLM_instr_files/RAG_instr.txt'))

        num_chunks_to_retrieve=get_config_value(self.inference_config, 'retrieval.num_chunks_to_retrieve', 20)
        num_cross_encoder_results=get_config_value(self.inference_config, 'retrieval.num_cross_encoder_results', 5)
       
        
        # Initialize the ModelInference class
        print("Initializing ModelInference...")
        model_inference = ModelInference(
            model_output_folder=model_output_folder,
            model_name=LLM_model_name,  # This should be passed as parameter or from config
            path_to_dataset=path_to_dataset,
            LLM_vendor_name=LLM_vendor_name,
            LLM_model_name=LLM_model_name,
            LLM_instr_filename=LLM_instr_filename,
            API_key_string=API_key_string
        )
        


        # Generate answers for each question
        dataset_with_answers = []
        
        for i, question in tqdm(enumerate(dataset["question"]),total=len(dataset["question"]),desc="Generating answers for questions"):
            current_dataset_entry={}
            current_dataset_entry["question"]=question
            
            
            try:
                answer = model_inference.query_model(
                    question=question,
                    num_embedding_results=num_chunks_to_retrieve,  # Number of chunks to retrieve initially
                    num_cross_encoder_results=num_cross_encoder_results  # Number of chunks after reranking
                )
                current_dataset_entry["answer"]=answer
                
            except Exception as e:
                print(f"Error processing question {i+1}: {e}")
                current_dataset_entry["answer"]=f"Error: Could not generate answer for this question. {str(e)}"
            dataset_with_answers.append(current_dataset_entry)
        
        print(f"✅ Generated answers for {len(dataset_with_answers)} questions")
        return dataset_with_answers

        
    def compare_answers(self):
        
        answers_pred=load_from_disk(self.eval_results_path)
        answers_true=load_from_disk(self.eval_dataset_path)

        combined_string=""

        # Fix: Use enumerate to get index and row properly
        for i, row in enumerate(tqdm(answers_true, desc="Comparing answers")):
            question=row["question"]
            answer_true=row["answer"]
            answer_pred=answers_pred[i]["answer"]
            if question != answers_pred[i]["question"]:
                print(f"Question: {question}")
                print(f"Question: {answers_pred[i]['question']}")
                print(f"❌ Question {i+1} is incorrect")
                input("check")
                continue
            
            combined_string+=f"Question: {question}\n"
            combined_string+=f"True Answer: {answer_true}\n"
            combined_string+=f"Answer from model: {answer_pred}\n"
            combined_string+="\n"

        print(combined_string)

        # create an LLM judge for this
        llm_instance=create_LLM_instance(self.llm_API_key, [], self.llm_model_name, self.llm_vendor_name)
        judge_answer=llm_instance.query_model(combined_string, self.judge_instructions_file_path)
        
        scores=[float(x) for x in judge_answer.split(",")]
        avg_score=sum(scores)/len(scores)

        max_score=max(scores)
        min_score=min(scores)
        print(f"Scores: {scores}")
        print(f"Average score: {sum(scores)/len(scores)}")
        print(f"Median score: {sorted(scores)[len(scores)//2]}")
        print(f"Max score: {max(scores)}")
        print(f"Min score: {min(scores)}")
        

        return avg_score, max_score, min_score

    def evaluate_model(self):
        
        self.load_dataset()

        # If not present, generate it:
        if not os.path.isdir(self.eval_dataset_path):
            self.generate_eval_data()
        else:
            print(f"✅ Evaluation dataset already exists at: {self.eval_dataset_path}")

        dataset=load_from_disk(self.eval_dataset_path)
        
        # generate answers from your RAG/RAG+SFT model

        dataset_with_answers=self.query_created_model(dataset)

        # save dataset with answers
        hf_dataset=Dataset.from_list(dataset_with_answers)
        hf_dataset.save_to_disk(self.eval_results_path)

        # compare answers with previously generated answers from GPT
        avg_score, max_score, min_score=self.compare_answers()

        return avg_score, max_score, min_score, self.dataset_config,self.model_config,self.inference_config

        
        
        
        
        


