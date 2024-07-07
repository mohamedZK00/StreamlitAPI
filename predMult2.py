import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Load the prediction model
working_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(working_dir, 'gbR94.sav')

with open(model_path, 'rb') as f:
    Model = pickle.load(f)

# Sidebar for navigation
with st.sidebar:
    options = ['Prediction Of Total Grades for All Students & Statisticas',
               'Predict Student Grades',
               'Classification of Student Grades']
    icons = ['pie','percent', 'mortarboard-fill']

    selected = option_menu('Multiple Student Grade Prediction System', options, icons=icons, default_index=0)

# Define data for random grades (assuming these are for the example data)
num_students = 1000
np.random.seed(42)
grades_month1 = np.random.randint(40, 75, num_students)
grades_month2 = np.random.randint(75, 101, num_students)
grades_month3 = np.random.randint(90, 101, num_students)

students_df = pd.DataFrame({
    'Student_ID': range(1, num_students + 1),
    'Grade_Month1': grades_month1,
    'Grade_Month2': grades_month2,
    'Grade_Month3': grades_month3
})

data_1 = students_df.drop('Student_ID', axis=1)

num_students = 1500
grades_month1 = np.random.randint(43, 49, num_students)
grades_month2 = np.random.randint(55, 80, num_students)
grades_month3 = np.random.randint(50, 65, num_students)

students_df_1 = pd.DataFrame({
    'Student_ID': range(1, num_students + 1),
    'Grade_Month1': grades_month1,
    'Grade_Month2': grades_month2,
    'Grade_Month3': grades_month3
})

data_2 = students_df_1.drop('Student_ID', axis=1)

num_students = 2000
grades_month1 = np.random.randint(40, 49, num_students)
grades_month2 = np.random.randint(45, 50, num_students)
grades_month3 = np.random.randint(38, 49, num_students)

students_df_2 = pd.DataFrame({
    'Student_ID': range(1, num_students + 1),
    'Grade_Month1': grades_month1,
    'Grade_Month2': grades_month2,
    'Grade_Month3': grades_month3
})

data_3 = students_df_2.drop('Student_ID', axis=1)

# Combine all data into one DataFrame
combined_data = pd.concat([data_1, data_2, data_3])

# Page for Prediction Of Total Grades for All Students & Statistics
if selected == 'Prediction Of Total Grades for All Students & Statisticas':
    st.title('Prediction Of Total Grades for All Students & Statisticas')

    st.info('Grades Prediction Passing & Failing')

    original_list = {
        '1-Grades for 1000 Students': data_1,
        '2-Grades for 1500 Students': data_2,
        '3-Grades for 2000 Students': data_3,
        '4-All Combined Grades': combined_data
    }

    select = st.selectbox('Select Your Set Of Grades', list(original_list.keys()))

    if st.button('Predict'):
        selected_data = original_list[select]

        selected_data_array = selected_data.values

        if np.any(np.isnan(selected_data_array)):
            st.error("Selected dataset contains NaN values. Please clean the data.")
        else:
            result = Model.predict(selected_data_array)

            pass_count = np.sum(result >= 50)
            fail_count = np.sum(result < 50)

            total_count = len(result)

            pass_percentage = (pass_count / total_count) * 100
            fail_percentage = (fail_count / total_count) * 100

            result_df = pd.DataFrame(result, columns=['Predicted Grades Of All'])

            st.write(f"Pass Percentage: {pass_percentage:.2f}%")
            st.write(f"Fail Percentage: {fail_percentage:.2f}%")

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            axes[0, 0].plot(result_df.index, result_df['Predicted Grades Of All'], marker='o', linestyle='-', color='#1f77b4')
            axes[0, 0].set_title('Predicted Grades')
            axes[0, 0].set_xlabel('Student ID')
            axes[0, 0].set_ylabel('Grades')

            axes[0, 1].bar(['Pass', 'Fail'], [pass_percentage, fail_percentage], color=['#1f77b4', '#ff7f0e'])
            axes[0, 1].set_title('Pass vs Fail Percentage')
            axes[0, 1].set_ylabel('Percentage')

            axes[1, 0].pie([pass_percentage, fail_percentage], labels=['Pass', 'Fail'], autopct='%1.1f%%', colors=['#1f77b4', '#ff7f0e'])
            axes[1, 0].set_title('Pass vs Fail Percentage')

            axes[1, 1].hist(result_df['Predicted Grades Of All'], bins=20, color='#2ca02c', alpha=0.75)
            axes[1, 1].set_title('Distribution of Predicted Grades')
            axes[1, 1].set_xlabel('Grades')
            axes[1, 1].set_ylabel('Frequency')

            fig.tight_layout()

            st.pyplot(fig)

# Page for Predict Student Grades
elif selected == 'Predict Student Grades':
    st.title('Predict Student Grades using ML')

    grade_month_1 = st.text_input('Degree of Month 1')
    grade_month_2 = st.text_input('Degree of Month 2')
    grade_month_3 = st.text_input('Degree of Month 3')

    if st.button('Grade Prediction'):
        grd_pred = Model.predict([[grade_month_1, grade_month_2, grade_month_3]])

        predicted_grade = {"grd_pred": round(grd_pred[0], 1)}
        max_grade = 100
        predicted_percentage = round((predicted_grade["grd_pred"] / max_grade) * 100, 1)
        formatted_result = "Student grade : {:.1f}%".format(predicted_percentage)

        st.success(formatted_result)

# Page for Classification of Student Grades
elif selected == 'Classification of Student Grades':
    st.title('Classification of Student Grades using ML')

    grade_month_1 = st.text_input('Degree of Month 1')
    grade_month_2 = st.text_input('Degree of Month 2')
    grade_month_3 = st.text_input('Degree of Month 3')

    if st.button('Grade Classification'):
        ls_pred = Model.predict([[grade_month_1, grade_month_2, grade_month_3]])

        def classify_grade(grade):
            if grade >= 90:
                return 'A'
            elif grade >= 80:
                return 'B'
            elif grade >= 70:
                return 'C'
            elif grade >= 60:
                return 'D'
            elif grade >= 50:
                return 'E'
            else:
                return 'F'

        classified_grades = [classify_grade(grade) for grade in ls_pred]
        st.success(classified_grades)

        def classify_grade_description(grade):
            if grade == 'A':
                return 'Excellent'
            elif grade == 'B':
                return 'Very Good'
            elif grade == 'C':
                return 'Good'
            elif grade == 'D':
                return 'Satisfactory'
            elif grade == 'E':
                return 'Sufficient'
            else:
                return 'Fail'

        clas_grades = [classify_grade_description(grade) for grade in classified_grades]
        st.success(clas_grades)
