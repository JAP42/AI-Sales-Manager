import whisperx
import json
import os
import requests
from mysql.connector import connect

def transcribe_and_diarize(audio_file):
    """
    Transcribes and diarizes an audio file using whisperx.

    Args:
        audio_file: Path to the audio file.

    Returns:
        A dictionary containing the transcription results.
    """
    model = whisperx.load_model("medium")  # Choose appropriate model size
    result = model.transcribe(audio_file)
    return result

def ollama_inference(prompt, model_name="openhermes2.5-mistral"):
    """
    Performs inference with the Ollama LLM.

    Args:
        prompt: The input text for the LLM.
        model_name: The name of the Ollama model to use.

    Returns:
        The response from the Ollama LLM.
    """
    url = "http://localhost:11434/api/chat"  # Replace with your Ollama server address
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # Raise an exception for bad status codes
    return response.json()["choices"][0]["message"]["content"]

def summarize_and_analyze_sentiment(transcript):
    """
    Summarizes the conversation and analyzes sentiment using Ollama.

    Args:
        transcript: The full transcription text.

    Returns:
        A tuple containing the summary and sentiment.
    """
    summary_prompt = f"Summarize the following conversation:\n\n{transcript}"
    summary = ollama_inference(summary_prompt)

    sentiment_prompt = f"Analyze the sentiment of the following conversation. Is it generally positive, negative, or neutral? Briefly explain your reasoning.\n\n{transcript}"
    sentiment_response = ollama_inference(sentiment_prompt)
    sentiment = sentiment_response.split("\n")[0]  # Extract the sentiment label

    return summary, sentiment

def save_to_database(filename, full_json, summary, sentiment):
    """
    Saves the transcription data to the SQL database.

    Args:
        filename: Name of the audio file.
        full_json: The full JSON output from whisperx.
        summary: The summarized conversation.
        sentiment: The sentiment of the conversation.
    """
    try:
        # Replace with your actual database credentials
        conn = connect(
            host="your_host",
            user="your_user",
            password="your_password",
            database="your_database"
        )
        cursor = conn.cursor()

        insert_query = """
        INSERT INTO conversations (filename, full_json, summary, sentiment) 
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(insert_query, (filename, json.dumps(full_json), summary, sentiment))
        conn.commit()
        print(f"Data for {filename} inserted successfully.")

    except Exception as error:
        print(f"Error inserting data: {error}")

    finally:
        if conn:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    audio_file = "path/to/your/audio.wav" 

    transcription_result = transcribe_and_diarize(audio_file)
    full_json = json.dumps(transcription_result)
    summary, sentiment = summarize_and_analyze_sentiment(transcription_result["segments"][0]["text"])
    save_to_database(os.path.basename(audio_file), full_json, summary, sentiment)
