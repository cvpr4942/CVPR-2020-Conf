Readme for the open set recognition code using the proposed method SP & KSP:

The genuine dataset is MNIST. The code is developed in colab, so MNIST training dataset is loaded from google drive. 
The following lines of code can be used by an interested reader 
to mount the Google Drive to Google Colab and download the data:

################################################################
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
link = 'put the link here' # The shareable link
fluff, id = link.split('=')
print (id) # Verify that you have everything after '='
downloaded = drive.CreateFile({'id':id}) 
downloaded.GetContentFile('Filename.csv')  
df3 = pd.read_csv('Filename.csv')
# Dataset is now stored in a Pandas Dataframe

##################################################################

Preprocessed omniglot data must be stored in google drive and loaded using the above piece of code.
One requires to upload the preprocessed omniglot file (included in the zip file) on the google drive and create the link to use it in the piece of code above (link section).
The mnist is loaded in the code as the dataloader is defined for colab and Keras. Thus, there is no need to download anything.
We have also provided the omniglot on github for interested readers.

A general summary of code blocks are provided as follows:

##### The function SP is designed to get a dataset and the number of required samples 

##### SP is called on each training class to select the best representatives.

##### A CNN is trained on the entire training dataset.

##### testing data consists of 10000 samples from Omniglot and the Mnist datasets which are set to have similar variance and size.

#### Next, the SOSIS algorithm is implemented using the selected samples. 
Determining whether a sample is from open set or not is based on the threshold level 
and whether its projection error on selected sample violates the threshold or not.


#### The macro-averaged F1-score, and ROC (varying the threshold level) are plotted.


The code and the data required are included. 