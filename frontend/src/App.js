import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [feedbackState, setFeedbackState] = useState(null); // { proposedAnswer, sessionId }
  const [feedbackText, setFeedbackText] = useState("");
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const exampleQuestions = [
    "Find the derivative of f(x) = x² + 3x - 5",
    "What is the integral of sin(x) dx?",
    "Solve the limit: lim(x→0) (sin x)/x",
    "Find the area under the curve y = x² from x = 0 to x = 3",
  ];

  const sendMessage = async (message) => {
    if (!message.trim()) return;

    const userMessage = { role: "user", content: message };
    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        message: message,
        session_id: sessionId,
      });

      const {
        response: agentResponse,
        session_id,
        status,
        proposed_answer,
      } = response.data;

      setSessionId(session_id);

      if (status === "waiting_for_feedback") {
        // Agent has prepared an answer and wants feedback
        setMessages((prev) => [
          ...prev,
          {
            role: "agent",
            content: agentResponse,
          },
        ]);

        setFeedbackState({
          proposedAnswer: proposed_answer,
          sessionId: session_id,
        });
      } else {
        // Regular response - conversation complete
        setMessages((prev) => [
          ...prev,
          {
            role: "agent",
            content: agentResponse,
          },
        ]);
        setFeedbackState(null);
      }
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          content:
            "Sorry, there was an error processing your request. Please try again.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const submitFeedback = async (feedback) => {
    if (!feedbackState) return;

    setIsLoading(true);
    setFeedbackText("");

    try {
      const response = await axios.post(`${API_BASE_URL}/feedback`, {
        session_id: feedbackState.sessionId,
        feedback: feedback,
        original_answer: feedbackState.proposedAnswer,
      });

      // Add feedback submission confirmation
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          content: `✅ Feedback submitted: "${feedback}". Thank you for helping improve the agent!`,
        },
      ]);

      // Add the final answer if available
      if (response.data.final_answer) {
        setMessages((prev) => [
          ...prev,
          {
            role: "agent",
            content: response.data.final_answer,
          },
        ]);
      }

      setFeedbackState(null);
    } catch (error) {
      console.error("Error submitting feedback:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          content: "Error submitting feedback. Please try again.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    sendMessage(inputMessage);
  };

  const handleExampleClick = (question) => {
    sendMessage(question);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="chat-container">
      {/* Header */}
      <div className="chat-header">
        <h1>🧮 Math Routing Agent</h1>
        <p>AI-powered math tutoring with knowledge base and web search</p>
      </div>

      {/* Messages */}
      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="empty-state">
            <h3>Welcome to Math Routing Agent!</h3>
            <p>
              Ask me any calculus question and I'll provide step-by-step
              solutions.
            </p>
            <div className="example-questions">
              <p>
                <strong>Try these examples:</strong>
              </p>
              {exampleQuestions.map((question, index) => (
                <div
                  key={index}
                  className="example-question"
                  onClick={() => handleExampleClick(question)}
                >
                  {question}
                </div>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <div key={index} className={`message ${message.role}`}>
              {message.role === "user" ? (
                message.content
              ) : (
                <ReactMarkdown>{message.content}</ReactMarkdown>
              )}
            </div>
          ))
        )}

        {/* Feedback Section */}
        {feedbackState && (
          <div className="feedback-section">
            <h4>📝 Please review my proposed answer:</h4>
            <div className="proposed-answer">
              <ReactMarkdown>{feedbackState.proposedAnswer}</ReactMarkdown>
            </div>
            <textarea
              className="feedback-input"
              placeholder="Please provide your feedback on this answer... (e.g., 'Good explanation', 'Missing step 3', 'Incorrect formula', etc.)"
              value={feedbackText}
              onChange={(e) => setFeedbackText(e.target.value)}
              disabled={isLoading}
            />
            <div className="feedback-buttons">
              <button
                className="btn btn-primary"
                onClick={() => submitFeedback(feedbackText)}
                disabled={isLoading || !feedbackText.trim()}
              >
                Submit Feedback
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => submitFeedback("Answer looks good")}
                disabled={isLoading}
              >
                Accept Answer
              </button>
            </div>
          </div>
        )}

        {/* Loading indicator */}
        {isLoading && (
          <div className="message agent">
            <div className="loading">
              <span>Thinking</span>
              <div className="loading-dots">
                <div className="loading-dot"></div>
                <div className="loading-dot"></div>
                <div className="loading-dot"></div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="input-container">
        <form onSubmit={handleSubmit} className="input-wrapper">
          <textarea
            className="message-input"
            placeholder="Ask a math question..."
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading}
            rows={1}
          />
          <button
            type="submit"
            className="send-button"
            disabled={isLoading || !inputMessage.trim()}
          >
            ⬆
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
