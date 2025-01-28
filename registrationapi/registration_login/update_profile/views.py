from django.shortcuts import render
from registration.models import StudentModel
from registration.serializer import StudentSerializer,serializers
from rest_framework import response, status
from rest_framework.views import APIView

class UpdateProfile(APIView):
    def post(self, request):
        user = request.user
        data = request.data
        user.first_name = data.get('first_name', user.first_name)
        user.last_name = data.get('last_name', user.last_name)

        user.save()
        return response.Response({'message': 'Profile updated successfully',
                                  "first_name":user.first_name,
                                  "last_name":user.last_name}
                                  , status=status.HTTP_200_OK)
    

    def patch(self, request):
        user = request.user
        serializer = StudentSerializer(user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return response.Response(serializer.data)
        return response.Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST) 
    
    def delete(self,request):
        user = request.user
        user.delete()
        return response(
            {'message': 'User deleted successfully'},
            status=status.HTTP_204_NO_CONTENT
        )