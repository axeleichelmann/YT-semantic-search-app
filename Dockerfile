# start from python base image
FROM python:3.10

# change working directory on docker image
WORKDIR /code

# add requirements file to docker image
COPY /requirements.txt /code/requirements.txt

# install python libraries
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# add python code to docker image
COPY ./app/ /code/app/

# specify default commands
CMD ["fastapi", "run", "app/main.py", "--port", "80"]
