# import sys
# if sys.version_info[0:2] != (3, 8):
#     raise Exception('Requires python 3.8')

from app import app
 
if __name__ == "__main__":
#   app.run(host="0.0.0.0", port=8080)
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
