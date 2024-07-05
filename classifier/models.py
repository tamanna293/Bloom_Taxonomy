from django.db import models

class Question(models.Model):
    question_text = models.TextField()
    bloom_taxonomy_level = models.CharField(max_length=100)

    def __str__(self):
        return self.question_text

class Prediction(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    predicted_label = models.CharField(max_length=100)
    predicted_label_encoded = models.IntegerField()

    def __str__(self):
        return f"Prediction for '{self.question}' - {self.predicted_label}"


