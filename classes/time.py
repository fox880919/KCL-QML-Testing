from datetime import datetime


class MyTimeHelper:

    def getTimeNow(self):

        # date = datetime.now().strftime('%Y-%m-%d')

        # time = datetime.now().strftime('%H:%M:%S')

        dateAndTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print('dateAndTime: ', dateAndTime)

        return dateAndTime


