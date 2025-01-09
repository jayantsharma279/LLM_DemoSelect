# LLM_DemoSelect
### Optimized Demonstration Selection for Few Shot Learning on GPT-4 for a Named Entity Recognition Task

#### This work aims to increase the F1 score for a NER task by strategically selecting demonstrations for in-context learning using an LLM. My approach was inspired by research on demonstration selection strategies that optimize the diversity and representation of entity labels. To address the imbalance in label distribution in the training dataset, I wrote a get_shuffled_chat_history function, which reordered and stratified the training examples to ensure a more balanced exposure to different entity types during few-shot learning. By incorporating this strategy, I reduced the standard deviation of label occurrences in the prompt examples, thereby increasing label diversity.

The files BERT_Baseline.ipynb contains the working code for a BERT model fine-tuned for the specific NER task. The input files contain train and validation (dev) datasets for the model, and the model reports the validation accuracy and F1 score. Hyperparameters can be adjusted but tuned for the best performance without a long run time. Running the code on a GPU is highly suggested. 

LLM_Demonstration_Selection.ipynb initially defines the functions used to call the OpenAI API for any model. Users can input their API key in the code block commented or select the particular model to call (GPT-4 used by default). The code uses a train dataset to display demonstrations for few-shot learning to the LLM, based on the number of 'shots' set as a parameter. 

### Demonstration Selection: 
The later part of the code contains the modified function for shuffling the demonstration input to LLM, using a statistical analysis of the dataset to ensure a diverse set of demonstrations unbiased to any particular label. It is recommended not to exceed the shots to more than 20, given the amount of computing time required to process the shuffle of the demonstrations for a huge dataset. 
