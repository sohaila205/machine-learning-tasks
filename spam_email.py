import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#read dataset
dataset = pd.read_csv("spambase.txt",  sep = ',', header= None )
print(dataset.head())
#rename columns
dataset.columns  = ["word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", 
                      "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet", 
                      "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will", 
                      "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free", 
                      "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit", 
                      "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp", 
                      "word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs", 
                      "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85", 
                      "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
                      "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re", 
                      "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(", 
                      "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_hash", "capital_run_length_average", 
                      "capital_run_length_longest", "capital_run_length_total", "spam"]

print(dataset.head())

X = dataset.drop("spam", axis = 1)
Y = dataset.spam.values.astype(int)

from sklearn.preprocessing import scale
X = scale(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 4)

model = SVC( C = 1)

model.fit(X_train, Y_train)
#predict
y_pred = model.predict(X_test)

from sklearn import metrics 
#accurcy
metrics.confusion_matrix(y_true = Y_test, y_pred=y_pred)

print("accuracy", metrics.accuracy_score(Y_test, y_pred))
