__author__ = 'KranthiDhanala'

import httplib2, json
from oauth2client import  file, client
from googleapiclient import discovery
from sklearn.metrics import classification_report
from string import ascii_uppercase
from googleapiclient.errors import HttpError
from collections import OrderedDict
import numpy as np
import pandas as pd

project_id = "ml-letters"
model_id = "apicall6"
labels = OrderedDict((ch, idx) for idx, ch in enumerate(ascii_uppercase, 1))

#authentication
def get_prediction_api():
    scope = ['https://www.googleapis.com/auth/prediction','https://www.googleapis.com/auth/devstorage.read_only']
    service_account=True
    details = file.Storage('oAuth2.json')
    credentials = details.get()
    if credentials is None or credentials.invalid: #check if new oAuth flow is needed
        if service_account:
            with open('ML-LetterClassification-6cd0b89dfc2a.json') as f:
                account = json.loads(f.read())
                email = account['client_email']
                key = account['private_key']
            credentials = client.SignedJwtAssertionCredentials(email, key, scope=scope)
            details.put(credentials)

    http = credentials.authorize(httplib2.Http())
    return discovery.build("prediction", "v1.6", http=http)

#get the trained model
def get_model(api):
    return api.trainedmodels().get(project=project_id, id=model_id).execute()

def main():
    #get access to api
    api = get_prediction_api()

    #get the trained model

    try:
        model = get_model(api)
    except HttpError as e:
        if e.resp.status == 404:
            print("Model does not exist yet.")
                #new model
            api.trainedmodels().insert(project=project_id, body={
                'id': model_id,
                'storageDataLocation': 'datasets-letter/train_data.csv',
                'modelType': 'CLASSIFICATION'
            }).execute()
            model = get_model(api)
        else:
            print(e)

    if model.get('trainingStatus') != 'DONE':
        print "Model is still training"
        exit()

    print "Model is trained and ready for predictions"

    pred = np.array([])

    #for each item in test file make predictions
    with open('test_data.csv') as f:
        for line in f:
            record = line.split(',')
            #make predictions
            prediction = api.trainedmodels().predict(project=project_id, id=model_id, body={
            'input': {
            'csvInstance': record
            },
            }).execute()
            label = prediction.get('outputLabel')
            pred = np.append(pred,int(label))
    # original classes of test data
    actual = pd.read_csv("test_pred.csv",header = None)

    pred = pred.astype(int)
    target_nbrs = range(1,27)
    target = [str(x) for x in target_nbrs]
    #pred has all the classification
    print(classification_report(actual[0].values, pred,target_names= target))

    return


if __name__ == "__main__":
    main()