from django.shortcuts import render
from registration.models import StudentModel
from rest_framework import response, status
from rest_framework.views import APIView

class StudentList(APIView):
    def get(self, request):
        students = StudentModel.objects.all()
        return response.Response({'students': students}, status=status.HTTP_200_OK)
    
   