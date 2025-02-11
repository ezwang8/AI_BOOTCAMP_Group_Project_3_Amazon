# AI_BOOTCAMP_Group_Project_3_Amazon
![image](Resources/Main_page_icon.jpg)

## Overview
This Project will be looking at creating an AI model which will help users find "The True rating for a Product". In this project we are looking at Camera as our area of focus.

## Table of Contents
1. [Introduction](#Introduction)
2. [About the Data](#About-the-Data)
3. [Cleaning Up the Data](#Cleaning-up-the-Data)
4. [Visualizing the Data](#Visualizing-the-Data)
5. [Creating the Model](#Creating-the-model)
6. [Conclusion](#Conclusion)

## Introduction
At this point in time, everything is being reviews products, services, businesses, restaurants, movies, books and even personal performance - essentially anything where people can provide feedback or critical analysis. The problem with this review system is that It can be manipulated by providing fake review, paid reviews etc. which would unnaturally increase or decrease the rating of a product. Our goal in this project in to create an AI model which would be able to differentiate real reviews from fake and at the end determine what is the actual rating of the product.

## About the Data
The data we have take is from [kaggle.com](#https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset/data) which is collected over a period of two decades from Amazon's iconic products (a.k.a Product Reviews). The first review goes back to 1995, millions of Amazon customers write reviews to express their views on the products they purchased. The data contains the following columns marketplace, customer_id, review_id, product_id, product_parent, product_title, product_category, star_rating, helpful_votes, total_votes, vine, verified_purchase, review_headline, review_body, review_data.

Below are a couple of things we can understand from the Customer Reviews

<p align="center">
  <img width="850" height="600" src="https://github.com/user-attachments/assets/3dbb0bcc-c81e-4fce-b031-1d7a03f9bf13" alt="Description" width="850">
</p>

Below graph tell us the correlation matrix between different columns. These columns are shown in the graph are out of 15 columns which have some what of a correlation.

<p align="center">
  <img width="850" height="600" src="https://github.com/user-attachments/assets/99d87e50-e170-4924-8ba0-83a866b033b6" alt="Description" width="850">
</p>

Below graph show the relationship between "Verified Purchases" VS "Un-Verified Purchases"

<p align="center">
  <img width="850" height="600" src="https://github.com/user-attachments/assets/ac7a1d81-7b84-453c-91ee-06646ad576f2" alt="Description" width="850">
</p>

## Cleaning Up the Data
Since this data contain customer experiences, they contain emoji's ('❤'), short form sentences and in some cases html code. So cleaning the data would require removing all the above mentioned items but also maintaining proper sentences. Below are the sequence of steps taken to clean up the data.
1. Combine columns "review_body", "review_headline" and "product_title" into single column called "review_combined". The Logic behind this approach is to clean all thress columns in a single column.
2. Create two DataFrame, one for HTML code words and another for emoji. This approach will let us understand which columns contain html code and which contains emoji, this is another way to get more insight into the data.

```python
review_http = reviews_data[reviews_data.review_combined.str.contains(r'^(?=.*/>)(?=.*http)')]
review_emoji = reviews_data[reviews_data.review_combined.str.contains('❤')]
```
3. Using Regular Expression and Lamda function clean the "review_combined" column

```python
reviews_data.loc[:,'cleaned_reviews'] = reviews_data['review_combined'].apply(lambda x: re.sub(r'<[^>]+>|\W|http\S+', ' ', x.lower()))
```
4. Using Gensim Preprocessing function we remove "Stop Words" from the column which does further cleaning of data.

Using Word Cloud we created two images which shows the different before we clean "Stop Words"

<p align="center">
  <img width="1000" height="600" src="https://github.com/user-attachments/assets/ee7e723b-2ea7-4aec-a48a-ae2174d75197" alt="Description" width="1000">
</p>

Word Cloud image after Cleaning Stop words

<p align="center">
  <img width="1000" height="600" src="https://github.com/user-attachments/assets/526d9fea-50a8-4d9a-b9af-9632b955ceda" alt="Description" width="1000">
</p>

## Visualizing the Data

After Cleaning is done, Now we can understand the data even more to find out what are the fequently used words, for this we created the following function

```python
def barplot_most_freq_words(attribute, num_of_words, label):
    words = ' '.join(reviews_data[attribute]).split()
    word_counts = Counter(words)
    
    # Get top 20 words and their counts
    top_n_words = word_counts.most_common(num_of_words)
    print(top_n_words)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=[word[1] for word in top_n_words], y=[word[0] for word in top_n_words], palette='pastel')
    plt.xlabel('Word Count')
    plt.ylabel('Word')
    plt.title(f'Top 20 Words in {label}')
    plt.show()
```
Below is the graph which shows the output of the function which we used to find top 20 words in uncleaned data

<p align="center">
  <img width="1000" height="600" src="https://github.com/user-attachments/assets/32f48934-3bc0-407b-97b7-c1346b083588" alt="Description" width="1000">
</p>

Below is the graph when functioned is used to against cleaned data

<p align="center">
  <img width="1000" height="600" src="https://github.com/user-attachments/assets/757852eb-1f40-4681-8fa2-7ac0096f2482" alt="Description" width="1000">
</p>

After this our team wanted to go DEEEEP! we wanted to know what are the common words being used for all the star ratings.

<p align="center">
  <img width="1000" height="600" src="https://github.com/user-attachments/assets/92b2b839-240c-4681-a77b-f14b8f3fe8bd" alt="Description" width="1000">
</p>

While this is one way of looking at data we want to know sequencing like which words are used in sequence. For this we used "ngrams" from "nltk.util" the function created as follows

```python
def plot_ngram(attribute, n):
    all_words = ' '.join(reviews_data[attribute]).split()
    n_grams = list(ngrams(all_words, n))
    n_gram_freq = FreqDist(n_grams)
    
    plt.figure(figsize=(10, 6))
    n_gram_freq.plot(20, cumulative=False)

    plt.show()
```

The result of this funtion when used against "Cleaned reviews"

<p align="center">
  <img width="1000" height="600" src="https://github.com/user-attachments/assets/e26177a8-018b-49c6-8941-f2ac98ba58a6" alt="Description" width="1000">
</p>

