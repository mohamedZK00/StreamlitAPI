#import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Prediction Model
#Model_1 = pickle.load(open('StreamlitAPI/mdel87%.PLK', 'rb'))



# فتح الملف بنمط 'rb' (قراءة بنمط ثنائي)
#with open('mdel87%.PLK', 'rb') as file:
  #  Model_1 = pickle.load(file)

#working_dir = os.path.dirname(os.path.abspath(__file__))
#Model_1 =pickle.load(open('Saved_model/mdel87%.PLK','rb'))


#Model_1 = pickle.load(open('GBRmodel.plk','rb'))
try:
    with open('GBRmodel.plk', 'rb') as file:
        Model = pickle.load(file)
except Exception as e:
    print("حدث خطأ أثناء فتح الملف:", e)


#with open('GBRmode.plk', 'rb') as f:
#    Model_1 = pickle.load(f)
# Classification Model
#Model_2 = pickle.load(open('B:\\ML-Streamlit\\GBRmodel_99%_3.PLK', 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Student Grade Prediction System',
                           ['Predict Student Grades',
                            'Classification of Student Grades'],
                           icons=['percent','mortarboard-fill'],
                           default_index=0)

# Predict student grades
if selected == 'Predict Student Grades':
    # Page title
    st.title('Predict Student Grades using ML')
    
    grade_month_1 = st.text_input('degree of Mo.1')
    grade_month_2 = st.text_input('degree of Mo.2') 
    grade_month_3 = st.text_input('degree of Mo.3') 
    
    #code for predicton 
    pred_Grd = ''
    
    #creating a button for prediction
    if st.button('Grade prediction'):
        grd_pred = Model.predict([[ grade_month_1 , grade_month_2 , grade_month_3 ]])
        
        predicted_grade = {"grd_pred": round(grd_pred[0], 1)}  
        max_grade=100
        predicted_percentage = round((predicted_grade["grd_pred"] / max_grade) * 100, 1)
        formatted_result = "Student grade : {:.1f}%".format(predicted_percentage)
        
        st.success(formatted_result)
    

# Classification of student grades
if selected == 'Classification of Student Grades':
    # Page title 
    st.title('Classification of Student Grades using ML')
    
    grade_month_1 = st.text_input('degree of Mo.1')
    grade_month_2 = st.text_input('degree of Mo.2') 
    grade_month_3 = st.text_input('degree of Mo.3') 
    
    #code for predicton 
    Cls_Grd = ''
    
    #creating a button for prediction
    if st.button('Grade Classification '):
       ls_pred = Model.predict([[ grade_month_1 , grade_month_2 , grade_month_3 ]])
       
       
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
       classified_grades = [classify_grade(grade) for grade in ls_pred ]
       st.success( classified_grades)
       def classify_grade(grade):
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
       clas_grades = [classify_grade(grade) for grade in classified_grades ]
       st.success( clas_grades)
        
        
        
    
