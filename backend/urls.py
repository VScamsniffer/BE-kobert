from django.contrib import admin
from django.urls import path
from uploadfile.views import AudioFileUploadView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("upload-audio/", AudioFileUploadView.as_view(), name="upload-audio"),
    # path("analyze-file/", AudioFileUploadView.as_view(), name="analyze-file"),
]
