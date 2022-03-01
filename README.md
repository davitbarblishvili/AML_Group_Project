Team: 
- Sahil Patel (sjp2232) 
- Davit Barblishvili(db3230)
- Mausam Agrawal (msa2213) 
- Safa Shaikh(ss6166)

---
# Problem Statement
---
 **Project Proposal**
> A person’s income is a result of many variables in the given person’s life including their career-field, industry-demand, experience, education, opportunity, age, etc. In this project, we want to explore and analyze some of these variables and understand their impact on people’s incomes through the power of machine learning models. <br>

> First, we will try to predict a person’s income (classify it as either above $50K or not) by developing mostly supervised machine learning models using some of the above mentioned variables. This part of the project will involve understanding the data, fitting and optimizing machine learning models (See List Below).<br>

> Second, we will compare and rank the trained machine learning models. This part of the project will involve analysis of the time-efficiency, space-efficiency, classification accuracy and explainability of the models. <br>

**Dataset:** Adult Census Income Dataset from UCI <br>
**Dataset Description:** 
  - **Timeline:** 1994 
  - **Source:** This dataset is a subset extracted from the US Census Bureau Database (which
                in turn is collected by the US government) Format: A simple .csv file <br>
  - **Metadata:** 
    - Rows: 
      - An entry for each person. 
    - Columns: <br>
      - ”age”: The person’s age in years (integer) 
      - ”workclass”: The person’s career sector (string*) 
      - ”fnlwgt”: Census Bureau’s weight factor (integer) - WE WILL IGNORE 
      - ”education”: The person’s education level (string*) 
      - ”education.num”: The person’s education level in years (integer) 
      - -”marital.status”: The person’s relationship status (string*) 
      - -”occupation”: The person’s profession (string*)
      - ”relationship”: The person’s relationship status description (string*) 
      - -”race”: The person’s race (string*)
      - ”sex”: The person’s sex (string*)
      - ”capital.gain”: The person’s record for total gain if positive(integer) 
      - ”capital.loss”: The person’s record for total gain if negative(integer)
      - ”hours.per.week”: The person’s avg hours worked in a week (integer)
      - ”native.country”: The person’s native country (integer)
      - ”income”: The person’s income in comparison to $50k (string*) <br>
      *The string columns can be interpreted as “categorical” variables <br>
      
      
**Proposed ML Techniques**
- Logistic regression - Baseline model for comparison
- SVM
- Random Forest
- KNN
- Naive Bayes
- 1DCNN
Exploratory Data Analysis
- Age Histogram (stacked for >50K and <=50K)
- Working Class Histogram (stacked for >50K and <=50K)
- Education Histogram (stacked for >50K and <=50K)
- Male vs Female and their income distribution.
