import sys
import logging

# Custom exception class that extends the base Exception class
class CustomException(Exception):
    
    # The constructor accepts the error message and error details
    def __init__(self, error_message: Exception, error_detail: sys):
        # Initialize the base class (Exception) with the error message
        super().__init__(error_message)
        
        # Get the detailed error message by calling the static method
        self.error_message = CustomException.get_detailed_error_message(error_message, error_detail)

    # Static method to create a detailed error message that includes file name, line number, and the error message
    @staticmethod
    def get_detailed_error_message(error: Exception, error_detail: sys):
        # Extract detailed error information from the traceback
        _, _, exc_tb = error_detail.exc_info()
        
        # Get the file name where the error occurred
        file_name = exc_tb.tb_frame.f_code.co_filename
        
        # Get the line number where the error occurred
        line_number = exc_tb.tb_lineno
        
        # Format and return a detailed error message
        error_message = f"Error occurred in script: {file_name} at line {line_number}: {str(error)}"
        return error_message

    # Override the __str__ method to return the detailed error message when the exception is raised
    def __str__(self):
        return self.error_message
