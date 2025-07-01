#############################################
Command to run:
1. docker info
2. docker pull qdrant/qdrant 
3. docker run -p 6333:6333 -v .:/qdrant/storage qdrant/qdrant


############################################################
If all libraries are install then move forward otherwise run this command pip install -r requirements.txt
Then open the terminal in pwd:
In main working folder qdrant-rag

python3 -m venv .venv 
source .venv/bin/activate 

###########################################################

Now, 
run the app_gradio.py using python3 app_gradio.py for frontend


