from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('vis1/', views.vis1, name='Vis1'),
    path('vis2/', views.vis2, name='Vis2'),
    path('vis3/', views.vis3, name='Vis3'),
    path('vis4/', views.vis4, name='Vis4'),
]