## How to get the website working

1.	Download latest version of Anaconda 
2.	Download the .yaml file from our project 
3.	Open Anaconda Prompt
4.	We will now create the environment to run the website. Type this into Anaconda Prompt `conda env create -f <path\to\file.yaml>`
    My one looked like this: `conda env create -f C:\Users\Michael\Downloads\file.yaml`
5.	Wait for the packages to install 
6.	Once that has finished its time to activate the environment. Type `conda activate IndustryProject`
7.	Wait for this to finish and once it has it will say IndustryProject on the left. To deactivate it you will need to type “deactivate” but do not do this yet. 
8.	Once activated, type this into Anaconda Prompt `python`
9.	Then type `import nltk`
10.	After that type `nltk.download(‘stopwords’)`
11.	Once that has finished, then type `nltk.downlaod(‘punkt’)`
12.	Once this is done, you have now got the necessary packages installed to run the website and the environment should be ready to use. 
13.	To run the website, you will need to make sure the `IndustryProject` environment is activated
14.	Make sure you have downloaded the website. If not, then do so.
15.	Unzip it to your preferred folder.
16.	Once that is done go back to Anaconda Prompt and type the path to the website. For me I typed `cd C:\Users\Michael\Downloads\Portal\django_website`. This will set the path to the website.
17.	After your path is set, it is now time to run the server. Type `python manage.py runserver`
18.	Once the server is running, copy the localhost which will look something like this `http://127.0.0.1:8000/`
19.	After you have copied this, paste it into your browser and it should be the working website. To deactivate the server, go back to Anaconda Prompt and press `ctrl+C` and this should stop it. 
Finish
To run the server again, you will need to activate the IndustryProject environment again with “activate IndustryProject” and path to the website if it isn’t done already and run the server with `python manage.py runserver` again.
