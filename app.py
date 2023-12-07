'''
Author: Bappy Ahmed
Email: entbappy73@gmail.com
Date: 2021-Dec-18
'''

import pickle
import streamlit as st
import numpy as np


st.header('Book Recommender System Using Machine Learning')
model = pickle.load(open('artifacts/model.pkl','rb'))
book_names = pickle.load(open('artifacts/book_names.pkl','rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl','rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl','rb'))


def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])

    for name in book_name[0]: 
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['image_url']
        poster_url.append(url)

    return poster_url



def recommend(book_name):
    book_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    poster_url = fetch_poster(suggestion)
    books = book_pivot.index[suggestion[0]]
    for j in books:
        book_list.append(j)
    return book_list, poster_url



selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)

if st.button('Recommend'):
    books, urls = recommend(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.subheader(books[1])
        st.image(urls[1])
    with col2:
        st.subheader(books[2])
        st.image(urls[2])
    with col3:
        st.subheader(books[3])
        st.image(urls[3])
    with col4:
        st.subheader(books[4])
        st.image(urls[4])
    with col5:
        st.subheader(books[5])
        st.image(urls[5])
