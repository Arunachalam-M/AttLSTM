# Attention Based LSTMs

Detect duplicate questions in Quora with interpretable models

Our goal is to identify the duplicate pairs of questions from the Quora dataset and classify them correctly. An Attention based LSTM provides interpretable models and extracts key information from the questions. Many recent papers have shown that Content-independent naive features called 'leakage features' are unreasonably predictive in datasets like the quora QP dataset. We observed the same with some minor differences in output probabilities if the questions are input in a different order. This was the inspiration to build an attention based model where every output decision can be attributed to a simple dot product similarity between certain key words extracted in each question, which can be visualized to examine and correct.

Model Architecture:

Attention Block for interpretation and summarization

![Attention Block for interpretation and summarization](https://github.com/Arunachalam-M/AttLSTM/blob/master/Architecture1.png)


Complete Model for Duplicate detection

![Complete Model for Duplicate detection](https://github.com/Arunachalam-M/AttLSTM/blob/master/Architecture2.png)


The primary purpose of this model was to give insights into the interpretability of the model and some amount of information extraction. The architecture is also designed to be more generalizable across many applications by avoiding leakage features. After 5 epochs of training, I got a trainingaccuracy of 84% in the duplicate detection problem. I also got a 80.5% accuracy in the test set. The results in Information extraction and summarization also seem promising. The observations and results regarding the same are provided below.

![Attention Results 1](https://github.com/Arunachalam-M/AttLSTM/blob/master/1.png)

In the above sentence "I made the mistake of searching his social media and now I think he is too happy for me. Any words to make me feel better?", the words 'mistake', 'searching', 'social', 'media', 'words', 'feel', 'better' are given attention over the threshold. So the above sentence will be summarized as 'mistake searching social media words feel better' which is a reasonable summarization of the request with little details left out.

![Attention Results 2](https://github.com/Arunachalam-M/AttLSTM/blob/master/2.png)

Similarly in the above example we see the sentence 'What is the most intelligent thing a kid has ever said to you?' being summarized as 'most intelligent thing kid ever said you' which is also a pretty accurate summarization of the sentence.

Aside from summarization, we can also see how the model identiﬁes duplicates by assigning attention in the following example.

![Attention Results 3](https://github.com/Arunachalam-M/AttLSTM/blob/master/3.png)

![Attention Results 4](https://github.com/Arunachalam-M/AttLSTM/blob/master/4.png)


This was a duplicate question pair. The model predicted it correctly assigning it a 78% chance of being duplicates. But it is interesting to note that in addition to all the key words in the sentence, it assigned attention to 'Where' in the ﬁrst question but didn't assign attention to 'What' in the second question. We can see that the second question can be summarized easily as 'Digital Marketing Course Beginners' without the 'What'. But 'where' is essential to signify the source of the digital marketing course as that was an important part of the intent of the question.

The model makes some mistakes for small questions. For eg. the question pair 'How is Israel ﬁghting ISIS?' and 'Why is Israel Fighting ISIS?' are both summarized as ”Israel ﬁghting ISIS” without distinguishing between How and Why. Here 4 words out of the 5 are essential to convey the complete meaning of the question. We used cross alignment to see which words do not have equivalents in the other sentence as shown below.

![Cross Alignment](https://github.com/Arunachalam-M/AttLSTM/blob/master/cross_alignment.png)

Ideally, we expected the unaligned words to have distributed attention across other words and this can be used to quantify the cross attention between sentences using cosine similarity. But most words had the highest alignment with prepositions since the underlying fasttext embeddings are based on contextual similarity. This has reduced the importance of cross alignment features and they had a very low weight in the ﬁnal linear layer determining the probability of duplicates.

For the Attention based LSTM, we generated 4 different features as below. The ﬁnal linear layer used these 4 features to determine the duplicate output probability. The coefﬁcients of these features are also provided beside them to indicate their relative importance in identifying duplicates. 

• Cosine Similarity of Attention weighted embeddings - 9.2 
• Bilinear layer output to identify differences in embeddings - (-6.8) 
• Cross Alignment Similarity of Question 1 with Question 2 - 0.58 
• Cross Alignment Similarity of Question 2 with Question 1 - 0.5 

The above weights show that the ﬁrst 2 features of weighted attention and difference identiﬁcation were signiﬁcant in identifying the duplicates compared to the cross alignment features which were of less use. This gives us an accuracy of 80.5%, Precision of 0.72, Recall of 0.76 and an F1 Score of 0.74. The ROC curve of the same is given below with an AUC of 0.88.

![ROC Attention](https://github.com/Arunachalam-M/AttLSTM/blob/master/ROC_Attention.png)

For the attention based model, we can conclude that the attention mechanism in LSTMs is interpretable and can be used for summarization purposes. Since any question can be used individually to compute the attention and the attention weighted embeddings, these vectors can be stored in a database and be used for simultaneous search for duplicates instead of combining the questions in pairs of 2 and identifying duplicates as used in other methods. The single head attention worked well and had a high predictive power but the cosine based cross alignment features did not work as intended. To further improve the model, we can try using a multi-head attention instead of cosine similarities to compute cross attention features although it would increase the time complexity of the attention mechanism to O(n2).

