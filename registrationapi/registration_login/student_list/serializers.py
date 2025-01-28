from rest_framework import serializers
from registration.models import StudentModel

class StudentSerializer(serializers.ModelSerializer):
    class Meta:
        model = StudentModel
        fields = ['first_name', 'last_name', 'email', 'is_active']
