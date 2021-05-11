# Website Fingerprinting by DNS over HTTPS
Traditional website fingerprinting is to analyze the encrypted traffic between a user and a web server, but the DoH based website fingerprinting is to analyze the encrypted traffic between a user and a DoH server. It can be treated as a supervised learning problem: collect samples of DoH traffic with correct label, and train a model, then deploy the model for prediction. 

**Data Collection**
>  The Alexa top 500 U.S. [websites](/collection/websites.txt) are used, and 10 samples of DoH traffic for each site are collected under the Kali VM environment (Firefox with DoH enabled) and Cloudflare DoH server mozilla.cloudflare-dns.com (104.16.248.249 and 104.16.249.249) from April 1st, 2021 to April 8th, 2021. 

**Data Preprocess**
> The 10 samples of 500 pcap files are pre-processed, and the TCP segment length and direction information are extracted from the pcap files. These information of each website are encoded into a JSON buffer and saved into a .json file for each sample. 

**Model Training**
> Stratified k-fold Cross Validation is used to evaluate the models, and k=5 is chosen for resampling the dataset. The uni-grams and bi-grams features are extracted from the dataset, and Random Forest classifier is used to train the model. The scikit-learn machine learning library in Python is used for model training and prediction. The average accuracy after training with each fold is 98%. 

**Model Prediction**
> The best pre-trained model is chosen and deployed into test environment (Kali VM). The real-time collected and processed data is saved into a temp file, and the name of the temp file is sent to the prediction module via pipe. Then the prediction module will read the real-time data from the temp file, make a prediction with the model, and convert the predicted label to a website domain name. 

This diagram covers the main software components. While training the model, the labelled traffic goes from left to right (top) and produces the models and results. While serving the model, the unknown traffic goes from left to right (bottom) and generates the prediction of websites. 
![Summary](/docs/summary.jpg)

To train the model, run **./doh_data_collect.sh ; ./doh_data_process.sh ; ./doh_data_classify.py train**

To serve the model, run **./doh_data_collect.sh | ./doh_data_process.sh | ./doh_data_classify.py serve**
