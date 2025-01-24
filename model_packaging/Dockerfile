FROM python:3.8
ENV DEBIAN_FRONTEND=noninteractive

#RUN apt-get clean && apt-get update -y && apt-get upgrade -y && pip install --upgrade pip

RUN pip3 install -U onnx flask

RUN pip3 install opencv-python 

# Install system dependencies  
RUN apt-get update && apt-get install -y \  
    libgl1-mesa-glx \  
    libglib2.0-0  

RUN pip3 install flask_cors
RUN pip3 install requests
RUN pip3 install Pillow  
RUN pip3 install scikit-learn  
RUN pip3 install scikit-image  






COPY ./rural /rural
COPY ./urban /urban
COPY ./static /static
COPY ./templates /templates
COPY ./*.py /
RUN chmod +x ./app.py
EXPOSE 80  
CMD ["python", "app.py"]  


#CMD ["/bin/bash", "-c", "python"," app.py --host='0.0.0.0' --debug=True"]
