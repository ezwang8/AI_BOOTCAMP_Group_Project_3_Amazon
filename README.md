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
  <img width="1000" height="450" src="https://github.com/user-attachments/assets/3dbb0bcc-c81e-4fce-b031-1d7a03f9bf13" alt="Description" width="1000">
</p>

Below image tell us the correlation matrix between different columns.

<p align="center">
  <img width="1000" height="450" src="" alt="Description" width="1000">
</p>

## Cleaning Up the Data
