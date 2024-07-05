from django.shortcuts import render
from .models import Question
from .nlp_classifier import classify_question, visu1, visu2, visu3, visu4
from django.http import HttpResponse

def index(request):
    if request.method == 'POST':
        question_text = request.POST['question']
        result = classify_question(question_text)
        return render(request, 'index.html', {'question_text': question_text, 'result': result})
    else:
        return render(request, 'index.html', {})

def vis1(request):
    visu1()
    return HttpResponse("Distribution of Questions through Bloom's Taxonomy Level")

def vis2(request):
    visu2()
    return HttpResponse("Number of Questions per Bloom's Taxonomy Level")

def vis3(request):
    visu3()
    return HttpResponse("Overall evaluation results for each BCLs class in the testing set")

def vis4(request):
    visu4()
    return HttpResponse("Results of weighted average over all BCLs classes in the testing set")







