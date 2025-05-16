from langchain_community.chat_message_histories import ChatMessageHistory

class TruncatedChatMessageHistory(ChatMessageHistory):
    def append(self, message):
        super().append(message)
        # Keep only the last 5 messages
        if len(self.messages) > 5:
            self.messages = self.messages[-5:]



# Dictionary to hold memory per session
message_histories = {}


def get_session_history(session_id):
    if session_id not in message_histories:
        message_histories[session_id] = TruncatedChatMessageHistory()
    return message_histories[session_id]