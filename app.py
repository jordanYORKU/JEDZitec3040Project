import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import main  
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Setting up page 
st.set_page_config(page_title="Party Check-In Dashboard", layout="wide")

@st.cache_data
def get_data():
    """
    Loads raw CSV, passes it through main.preprocess_data, 
    and returns a clean DataFrame.
    """
    try:
        with open('partydata.csv', 'r') as f:
            reader = csv.reader(f)
            header = next(reader) 
            raw_rows = list(reader)
            
        processed_rows = main.preprocess_data(raw_rows)
        
        return pd.DataFrame(processed_rows, columns=header)
    except FileNotFoundError:
        st.error("'partydata.csv' not found. Please ensure it is in the same folder.")
        return pd.DataFrame()

def to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else x

# Loading data
df = get_data()

if df.empty:
    st.stop()


df['Attended'] = (df['Checked In'] == 'Yes').astype(int)

feature_cols = ['Age', 'Gender', 'Ticket Type', 'Buyer Pays', 'Predicted Name Origin']
target_col = 'Attended'

#### MODEL TRAINING ####
st.title("Event Dashboard")

#Sidebar for testing split sizes
test_size = st.sidebar.slider("Test Split Size", 0.1, 0.4, 0.2)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_cols], df[target_col], 
    test_size=test_size, stratify=df[target_col], random_state=42
)

#Pipeline
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), ['Buyer Pays']),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ['Age', 'Gender', 'Ticket Type', 'Predicted Name Origin'])
])

models = {
    "Logistic Regression": Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000))]),
    "Decision Tree": Pipeline([("pre", preprocessor), ("clf", DecisionTreeClassifier(max_depth=5))]),
    "Naive Bayes": Pipeline([("pre", preprocessor), ("dense", FunctionTransformer(to_dense, accept_sparse=True)), ("clf", GaussianNB())])
}

#Train all models
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model


#GUI Tabs
tab1, tab2, tab3 = st.tabs(["Model Performance", "Interactive Prediction", "Cohort Exploration"])

#Model Performance Tab
with tab1:
    st.header("Evaluation Accuracy")
    
    # Displays accuracy for each model
    cols = st.columns(3)
    for i, (name, model) in enumerate(trained_models.items()):
        acc = model.score(X_test, y_test)
        cols[i].metric(name, f"{acc:.1%}")

    st.divider()
    c1, c2 = st.columns(2)
    
    #Confusion Matrix 
    with c1:
        st.subheader("Confusion Matrix")
        sel_viz = st.selectbox("Select Model", list(trained_models.keys()))
        confmat = confusion_matrix(y_test, trained_models[sel_viz].predict(X_test))
        
        fig_confmat, ax = plt.subplots(figsize=(5, 4))
        cax = ax.imshow(confmat, cmap='Blues')
        fig_confmat.colorbar(cax)
        
        for i in range(2):
            for j in range(2):
                ax.text(j, i, confmat[i, j], ha="center", va="center", color="black" if confmat[i, j] < confmat.max()/2 else "white")
        
        ax.set(xticks=[0,1], yticks=[0,1], xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], 
               xlabel="Predicted", ylabel="Actual", title=f"{sel_viz} Matrix")
        st.pyplot(fig_confmat)

    #Feature Importance
    with c2:
        st.subheader("Feature Importance")
        tmodel = trained_models[sel_viz]
        
        try:
            #Extract names
            cat_names = tmodel.named_steps['pre'].transformers_[1][1].get_feature_names_out()
            all_names = ['Price_Tier'] + list(cat_names)
            
            #Extract importance scores
            if sel_viz == "Logistic Regression":
                scores = tmodel.named_steps['clf'].coef_[0]
            else:
                scores = tmodel.named_steps['clf'].feature_importances_
            
            #Plot
            feat_df = pd.DataFrame({'Feature': all_names, 'Score': scores})
            feat_df = feat_df.sort_values(by='Score', key=abs, ascending=True).tail(15) # Top 15
            
            fig_imp, ax = plt.subplots(figsize=(5, 4))
            ax.barh(feat_df['Feature'], feat_df['Score'], color='#4CAF50')
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            ax.set_title("Top 15 Impacting Features")
            st.pyplot(fig_imp)
        except Exception as e:
            st.info("Feature importance not available for this model configuration.")

#Attendance Predictor Tab
with tab2:
    st.header("Predict Guest Attendance")
    
    with st.form("pred_form"):
        c1, c2 = st.columns(2)
        # Raw inputs 
        age_in = c1.number_input("Age", 18, 99, 21)
        gender_in = c1.selectbox("Gender", ["Male", "Female"])
        ticket_in = c2.selectbox("Ticket Type", ["General Admission", "VIP", "Early Bird", "Jedz Team", "Secret"])
        price_in = c2.number_input("Price Paid", 0.0, 100.0, 20.0)
        origin_in = st.selectbox("Name Origin", ["English", "Asian", "European"])
        model_choice = st.selectbox("Choose Model", list(trained_models.keys()))
        
        submit = st.form_submit_button("Predict")
    
    if submit:

        raw_row = ["0", str(age_in), gender_in, "?", ticket_in, price_in, origin_in]
        
        processed_row = main.preprocess_data([raw_row])[0]
        
        input_df = pd.DataFrame([processed_row], columns=['Index', 'Age', 'Gender', 'Checked In', 'Ticket Type', 'Buyer Pays', 'Predicted Name Origin'])
        #Predict
        modelp = trained_models[model_choice]
        prob = modelp.predict_proba(input_df[feature_cols])[0]
        
        #Display prediction
        st.divider()
        if prob[1] > 0.5:
            st.success(f"Likely to Check In ({prob[1]*100:.1f}%)")
        else:
            st.error(f"Unlikely to Check In ({prob[0]*100:.1f}%)")
            
        st.caption(f"Age binned to '{processed_row[1]}', Ticket grouped to '{processed_row[4]}', Price Tier '{processed_row[5]}'.")

#Explore Cohorts Tab
with tab3:
    st.header("Cohort Exploration")
    
    # Filter Controls
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_ticket = st.multiselect("Filter by Ticket Group", df['Ticket Type'].unique(), default=df['Ticket Type'].unique())
    with col_f2:
        filter_checkin = st.multiselect("Filter by Actual Status", ["Yes", "No"], default=["Yes", "No"])
    
    filtered_df = df[df['Ticket Type'].isin(filter_ticket)]
    filtered_df = filtered_df[filtered_df['Checked In'].isin(filter_checkin)]
    
    # Show Data
    st.dataframe(filtered_df)
    
    # Stats
    st.markdown(f"**Showing {len(filtered_df)} rows** out of {len(df)} total.")
    
    if not filtered_df.empty:
        checkin_rate = filtered_df['Attended'].mean()
        st.info(f"Check-In Rate for this selection: **{checkin_rate*100:.1f}%**")