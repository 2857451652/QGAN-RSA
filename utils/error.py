# self defined error
class selfDefinedError(Exception):
    def __init__(self, ErrorInfo):
        self.ErrorInfo = ErrorInfo

    def __str__(self):
        return self.ErrorInfo