from django.urls import path, re_path
from . import views

urlpatterns = [
    path('', views.home, name='home-page'),
    path('aiEngine/', views.aiEngine, name='ai-Engine'),
    path('contactUs/', views.contactUs, name='contact-us'),
	re_path(r'^.*submit', views.submit), #Added this line to match a URL with a 'submit' string in it
]

