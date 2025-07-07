# langProBe Benchmark Descriptions

This document provides detailed descriptions of all benchmarks available in the langProBe framework.

## Agent Benchmarks

### AppWorld
- **Description**: Application-based task execution benchmark
- **Purpose**: Tests an AI's ability to interact with and complete tasks in application interfaces
- **Dataset**: Custom dataset of application-based tasks
- **Evaluation**: Success rate in completing the specified tasks

## Non-Agent Benchmarks

### HeartDisease
- **Description**: Heart disease prediction using medical data
- **Purpose**: Tests an AI's ability to diagnose heart disease based on patient medical information
- **Dataset**: Uses the `buio/heart-disease` dataset with features like age, sex, chest pain type, blood pressure, etc.
- **Evaluation**: Accuracy in predicting presence of heart disease (yes/no)

### hover (HoVer)
- **Description**: Fact verification using Wikipedia evidence
- **Purpose**: Tests an AI's ability to verify claims using multiple evidence sources
- **Dataset**: Uses the `hover` dataset, focusing on 3-hop examples that require multi-step reasoning
- **Evaluation**: Accuracy in retrieving relevant documents that support or refute claims

### judgebench
- **Description**: Response quality evaluation and comparison
- **Purpose**: Tests an AI's ability to compare and evaluate the quality of different responses to the same question
- **Dataset**: Uses the `ScalerLab/JudgeBench` dataset with pairs of responses to compare
- **Evaluation**: Accuracy in selecting the better response between two options (A>B or B>A)

### MMLU (Massive Multitask Language Understanding)
- **Description**: Multiple choice questions across academic subjects
- **Purpose**: Tests an AI's knowledge across a wide range of academic disciplines
- **Dataset**: Uses the `cais/mmlu` dataset covering subjects like mathematics, history, law, medicine, etc.
- **Evaluation**: Accuracy in answering multiple-choice questions (A, B, C, D)

### gsm8k (Grade School Math 8K)
- **Description**: Mathematical reasoning and problem solving
- **Purpose**: Tests an AI's ability to solve grade-school level math word problems
- **Dataset**: Uses the `gsm8k` dataset with math word problems
- **Evaluation**: Accuracy in calculating the correct numerical answer

### MATH
- **Description**: Advanced mathematical problem solving
- **Purpose**: Tests an AI's ability to solve complex mathematical problems
- **Dataset**: Uses the `lighteval/MATH` dataset with advanced math problems
- **Evaluation**: Accuracy in providing the correct solution

### humaneval
- **Description**: Python code generation from natural language descriptions
- **Purpose**: Tests an AI's ability to write functional code based on problem descriptions
- **Dataset**: Uses the HumanEval dataset with programming tasks
- **Evaluation**: Functional correctness of the generated code

### Iris
- **Description**: Classification of iris flower species
- **Purpose**: Tests an AI's ability to classify items based on numerical features
- **Dataset**: Uses the classic `hitorilabs/iris` dataset with measurements of iris flowers
- **Evaluation**: Accuracy in classifying iris species (setosa, versicolor, virginica)

### IReRa (Information Retrieval and Reasoning)
- **Description**: Information retrieval and reasoning tasks
- **Purpose**: Tests an AI's ability to retrieve relevant information and reason over it
- **Dataset**: Custom dataset focused on information retrieval challenges
- **Evaluation**: Accuracy in retrieving relevant information and drawing correct conclusions

### hotpotQA
- **Description**: Multi-hop question answering with reasoning
- **Purpose**: Tests an AI's ability to answer questions that require integrating information from multiple sources
- **Dataset**: Uses the HotpotQA dataset with questions requiring multi-step reasoning
- **Evaluation**: Accuracy in answering questions that require connecting multiple facts

### hotpotQA_conditional
- **Description**: Conditional multi-hop question answering
- **Purpose**: Extension of hotpotQA with conditional reasoning requirements
- **Dataset**: Modified HotpotQA dataset with conditional reasoning challenges
- **Evaluation**: Accuracy in answering questions with conditional reasoning

### RAGQAArenaTech
- **Description**: Retrieval-augmented question answering
- **Purpose**: Tests an AI's ability to answer questions by retrieving and using external knowledge
- **Dataset**: Custom dataset focused on technical questions requiring retrieval
- **Evaluation**: Accuracy in answering questions using retrieved information

### swebenchAnnotation
- **Description**: Software engineering annotation tasks
- **Purpose**: Tests an AI's ability to understand and annotate code
- **Dataset**: Uses the SweBench dataset with code annotation tasks
- **Evaluation**: Accuracy in correctly annotating code segments

### scone (Sequential CONtext understanding and rEasoning)
- **Description**: Sequential context understanding and reasoning
- **Purpose**: Tests an AI's ability to reason over sequential information
- **Dataset**: Uses the ScoNe dataset with sequential reasoning tasks
- **Evaluation**: Accuracy in understanding and reasoning about sequential information

### MedQA
- **Description**: Medical question answering
- **Purpose**: Tests an AI's knowledge of medical concepts and terminology
- **Dataset**: Uses the `bigbio/med_qa` dataset with medical questions
- **Evaluation**: Accuracy in answering medical questions

### PubMedQA
- **Description**: Biomedical literature question answering
- **Purpose**: Tests an AI's ability to answer questions based on biomedical literature
- **Dataset**: Uses the PubMedQA dataset with questions about biomedical research
- **Evaluation**: Accuracy in answering questions about biomedical literature

### MedMCQA
- **Description**: Medical multiple choice question answering
- **Purpose**: Tests an AI's medical knowledge through multiple choice questions
- **Dataset**: Uses the `openlifescienceai/medmcqa` dataset with medical MCQs
- **Evaluation**: Accuracy in answering medical multiple choice questions 