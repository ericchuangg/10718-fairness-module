from FairClassification.Algorithms.utils import getStats
import pandas as pd
from FairClassification.Algorithms.utils import getStats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

starting_date = pd.to_datetime("2008/01/01")
training_set_length = pd.Timedelta("360days")
validation_set_length = pd.Timedelta("120days")
model_retrain_interval = pd.Timedelta("360days")

training_set_start = starting_date 
training_set_end = training_set_start + training_set_length
validation_set_end = training_set_end + validation_set_length
  
def random_forest(train_df, train_labels, val_df):
  scaler = MinMaxScaler()
  scaler.fit(train_df)
  X = scaler.transform(train_df)
  X_val = scaler.transform(val_df)
  y = train_labels

  clf = RandomForestClassifier(random_state=0).fit(X, y)
  probs = clf.predict_proba(X)

  train_df['logistic_prob'] = probs[:, 0]

  probs_val = clf.predict_proba(X_val)
  val_df['logistic_prob'] = probs_val[:, 0]

  return train_df, val_df

donations = pd.read_csv('data/donations.csv')
essays = pd.read_csv('data/essays.csv')
outcomes = pd.read_csv('data/outcomes.csv')
projects = pd.read_csv('data/projects.csv')
resources = pd.read_csv('data/resources.csv')

projects_donations = projects.merge(donations[["donationid", "projectid", "donation_timestamp", "donation_to_project"]], on="projectid")

new_donations = donations.merge(projects[["date_posted", "projectid"]], on="projectid", how="left")

new_donations["donation_to_project"] = new_donations["donation_to_project"] * (
    (pd.to_datetime(new_donations["donation_timestamp"]) - 
     pd.to_datetime(new_donations["date_posted"])) < pd.Timedelta("120day")).astype(float)

joined = projects.join(new_donations.groupby("projectid")["donation_to_project"].sum(), on="projectid", how="left")

joined = joined.fillna(0)

joined["fraction_funded"] = joined["donation_to_project"] / joined["total_price_excluding_optional_support"]
joined["fully_funded"] = 0
joined.loc[joined["fraction_funded"] >= 1, "fully_funded"] = 1

joined["date_posted"] = pd.to_datetime(joined["date_posted"])

high_priority_features = ["teacher_teach_for_america", "teacher_ny_teaching_fellow", "primary_focus_subject", 
                          "primary_focus_area", "resource_type", "poverty_level", 
                          "total_price_excluding_optional_support", "students_reached",
                          "historical_teacher_success_rate",
                          "historical_school_success_rate",
                          "historical_zip_success_rate",
                          "funding_one_week"]

historical_teacher_success_rate = joined[(joined['date_posted'] >= training_set_start) & (joined['date_posted'] < training_set_end)]
historical_teacher_success_rate = historical_teacher_success_rate.groupby("teacher_acctid")["fully_funded"].mean()
joined_new = joined.merge(historical_teacher_success_rate.rename('historical_teacher_success_rate').reset_index(), on="teacher_acctid", how="left")
joined_new = joined_new.fillna(0)

historical_school_success_rate = joined[(joined['date_posted'] >= training_set_start) & (joined['date_posted'] < training_set_end)]
historical_school_success_rate = historical_school_success_rate.groupby("schoolid")["fully_funded"].mean()
joined_new = joined_new.merge(historical_school_success_rate.rename('historical_school_success_rate').reset_index(), on="schoolid", how="left")
joined_new = joined_new.fillna(0)

historical_zip_success_rate = joined[(joined['date_posted'] >= training_set_start) & (joined['date_posted'] < training_set_end)]
historical_zip_success_rate = historical_zip_success_rate.groupby("school_zip")["fully_funded"].mean()
joined_new = joined_new.merge(historical_zip_success_rate.rename('historical_zip_success_rate').reset_index(), on="school_zip", how="left")
joined_new = joined_new.fillna(0)

funding_one_week = projects_donations[pd.to_datetime(projects_donations["date_posted"]) >= pd.to_datetime(projects_donations["donation_timestamp"]) - pd.Timedelta("7 days")]
funding_one_week = funding_one_week.groupby("projectid")["donation_to_project"].sum()
joined_new = joined_new.merge(funding_one_week.rename("funding_one_week").reset_index(), on="projectid", how="left")
joined_new = joined_new.fillna(0)

starting_date = pd.to_datetime("2008/01/01")

training_set_length = pd.Timedelta("360days")
validation_set_length = pd.Timedelta("120days")

model_retrain_interval = pd.Timedelta("360days")

records_df = []

high_priority_df = joined_new[high_priority_features]
high_priority_df = pd.get_dummies(high_priority_df, columns = ["teacher_teach_for_america", "teacher_ny_teaching_fellow", "primary_focus_subject", 
                            "primary_focus_area", "resource_type", "poverty_level"])

training_set = high_priority_df[(joined_new['date_posted'] >= training_set_start) & 
                        (joined_new['date_posted'] < training_set_end)]

validation_set = high_priority_df[(joined_new['date_posted'] >= training_set_end) & 
                        (joined_new['date_posted'] < validation_set_end)]

print("training_set_start:", training_set_start,  
        "training_set_end:", training_set_end, 
        "start_date_for_labels:", training_set_start + pd.Timedelta("120days"), 
        "end_date_for_labels:", training_set_end + pd.Timedelta("120days"))

print("validation_set_start:", training_set_end,  
        "validation_set_end:", validation_set_end, 
        "start_date_for_labels:", training_set_end + pd.Timedelta("120days"), 
        "end_date_for_labels:", validation_set_end + pd.Timedelta("120days"))

training_labels = joined_new[(joined_new['date_posted'] >= training_set_start) & 
                        (joined_new['date_posted'] < training_set_end)]['fully_funded']
validation_labels = joined_new[(joined_new['date_posted'] >= training_set_end) & 
                        (joined_new['date_posted'] < validation_set_end)]['fully_funded'] 

baseline = "random_forest"
training_set, validation_set = random_forest(training_set, training_labels, validation_set)
percentile = 0.1

split_value = training_set[baseline].quantile(q=1 - percentile)
split_failure_rate_val = validation_labels[validation_set[baseline] > split_value].mean()
precision_val = 1 - split_failure_rate_val
subsampled_val = validation_set[validation_labels != 1]
subsampled_val["true_positive"] = (subsampled_val[baseline] > split_value).astype(float)
recall_val = subsampled_val["true_positive"].mean()

print(i, baseline, percentile, 1 - validation_labels.mean(), precision_val, recall_val)

