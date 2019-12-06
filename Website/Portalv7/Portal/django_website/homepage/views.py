from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

def home(request):
    return render(request, 'homepage/home.html')

def aiEngine(request):
    return render(request, 'homepage/aiEngine.html')

def contactUs(request):
    return render(request, 'homepage/contactUs.html')

#This handles submission to the AI engine
def submit(request):
	# checks if POST request and that the form has the name of the element with the data we want (taname)
	if request.method == 'POST' and 'taname' in request.POST:

		# import function to run
		# from TestRun import testfunction
		from TestDjangoScript import text_preprocessing

		# call function
		# output = testfunction(request.POST.get('taname', ''))
		output = text_preprocessing(request.POST.get('taname', ''))
		# this is the style of the output Div tag
		style = "font-size: 1.5em; text-align: center; font-weight: bold;"

	# return user to required page
	#return HttpResponse(aiEngine)
	return render(request, 'homepage/aiEngine.html', {'output_style': style, 'output': output})