class speech_recognition():
    def __init__(self):
        pass
    def __call__(self):
        def list_microphones():
            mic_list = sr.Microphone.list_microphone_names()
            for i, mic in enumerate(mic_list):
            print(f"Microphone {i + 1}: {mic}")

        list_microphones()
        # Replace 'your_microphone_name' with the actual name of your microphone
        microphone_name = 0
        
        recognizer = sr.Recognizer()
    
        print("Before recording...")
    
        # Specify the source using the microphone name
        with sr.Microphone(device_index=microphone_name) as source:
            print("Say something:")
            audio = recognizer.listen(source)
    
        print("After recording...")
    
        try:
            Text = recognizer.recognize_google(audio, language="zh-TW")
            print("You said:", Text)
            return Text
        except recognizer.UnknowValueError:
            Text = "無法翻譯"
            return Text
        except sr.RequestError as e:
            Text = "無法翻譯{0}".format(e)
            return Text

class pdf2text():
    def __init__(self):
        pass
    def __call__(self, pdf_file):
        pass


class text2vector():
    def __init__(self):
        pass
    def __call__(self, text):
        pass


class cosine_sim():
    def __init__(self):
        pass
    def __call__(self, vector_from_table, vector_from_keyword):
        pass

def main(keyword, pdf_file):
    pass
    # return table

if __name__ == "__main__":
    main()
