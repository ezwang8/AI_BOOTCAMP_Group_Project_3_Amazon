# AI_BOOTCAMP_Group_Project_3_Amazon
![image](Resources/Main_page_icon.jpg)

## Overview
This Project will be looking at creating an AI model which will help users find "The True rating for a Product". In this project we are looking at Camera as our area of focus.

## Table of Contents
1. [Introduction](#Introduction)
2. [About the Data](#About-the-Data)
3. [Cleaning Up the Data](#Cleaning-up-the-Data)
4. [Visualizing the Data](#Visualizing-the-Data)
5. [Spam Detection](#Spam-Detection)
6. [Creating the Model](#Creating-the-model)
7. [Conclusion](#Conclusion)

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

```python
def worl_cloud(attribute, label):
  all_reviews_combined = " ".join(reviews_data[attribute])
  wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews_combined)
  plt.figure(figsize=(10, 6))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis('off')
  plt.title(f'Word Cloud for {label}')
  plt.show()
```


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

## Spam Detection
Spam is a major problem in the current world we are living in. We see SPAM in emails, phone calls, messages etc. Why can't we see spam reviews. Below are the steps we have gone through on how we came to detect SPAM in our reviews.

### 1. Spam Words :
   We came up with everyday words that co-inside with what spam might be and tried to find out if our data contains these works.
```python
spam_related_words = [
    "scam", "fake", "bot", "fraudulent", "ripoff", "misleading",
    "counterfeit", "false", "bait", "switch", "waste", "useless"
]
```
### 2. Too Few Words : 
  If the review contains too little charactors which might be viewed as a spam review. We have decided any reviews with less than 5 charactors     is a spam review.
### 3. Too Many Reviews from the same Customer too Soon:
  If the same customer is leaving multiple reviews on the same day, we decided that might be a spam review. No one would have the time to leave more than couple reviews a day, but the data shows customers who have written more reviews on the same day. Below code is able to findout which customers are doing this.
  ```python
# Flag customers who posted 10 or more fast reviews in a single day
spam_fast_customers = fast_reviews_per_day[fast_reviews_per_day["fast_review_count"] >= 10]["customer_id"].unique()

# Add a new spam column for fast reviewers
df["spam_fast_reviews"] = df["customer_id"].apply(lambda x: 1 if x in spam_fast_customers else 0)

# Check how many customers were flagged
print(df["spam_fast_reviews"].value_counts())

spam_fast_reviews
0    9892
1     108
Name: count, dtype: int64
```
### 4. Repetitive Reviews by the Same Customer
  Some Customers leave generic reviews, but same customer leaving similar reviews to different products can be concluded as spam reviews. We have setup a threshold of 90% similarity. We looked at each customer's reviews and compared it to other reviews done by the same customer. Below is some of the output we found where customers are leaving similar reviews.

|   | customer_id |                                          review_1 |                                          review_2 | similarity_score |
|--:|------------:|--------------------------------------------------:|--------------------------------------------------:|-----------------:|
| 0 |      109766 |                                 Item as described |                                 Item as described |         1.000000 |
| 1 |      111870 |                                              Good |                                              Good |         1.000000 |
| 2 |      123943 | The video not clear.<br />The sound is bad and... | The video not clear.<br />The sound is bad and... |         0.926773 |
| 3 |      137226 |                                            great! |                                            great! |         1.000000 |
| 4 |      154445 |                                             great |                                             great |         1.000000 |

### 5. Review with URLs
Reviews that contain excessive or irrelevant URLs are generally considered spam, as they are often used as a tactic to artificailly boost product ratings in search engines by manupilating link profiles. These type of reviews are considered as SPAM industry wide. we used a logic of "if a review_body has URLs, or any html code that is spam". We created a new column to give the review 0 or 1 depending if that review is spam or not.


