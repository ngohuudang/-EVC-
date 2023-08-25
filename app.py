from flask import Flask, render_template, send_file
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property

from flask_restx import Api, Resource, reqparse
from pydub import AudioSegment

from werkzeug.datastructures import FileStorage
import io

app = Flask(__name__)
api = Api(app, version='1.0', title='Audio Merge API', description='API for merging audio files')
# ns = api.namespace('audio', description='Audio operations')
# Define the request parser for file uploads
upload_parser = api.parser()
upload_parser.add_argument('audio1', location='files', type=FileStorage, required=True)
upload_parser.add_argument('audio2', location='files', type=FileStorage, required=True)

# upload_parser.add_argument('files', location='files', type=FileStorage, action='append', required=True, help='List of audio files to merge')

# # Audio model for response
# audio_model = api.model('Audio', {
#     'merged_audio': fields.String(description='Download link for merged audio file')
# })

@app.route('/')
def index():
    return render_template('index.html')

@api.route('/merge') 
class MergeAudio(Resource):
    @api.expect(upload_parser)
    # @api.marshal_with(audio_model)
    # def __init__(self):
    #     self.upload_parser = api.parser()
    #     self.upload_parser.add_argument('audio1', location='files', type=FileStorage, required=True)
    #     self.upload_parser.add_argument('audio2', location='files', type=FileStorage, required=True)
    #     super(MergeAudio, self).__init__()
    def post(self):
        try:
            args = upload_parser.parse_args()
            audio1 = args['audio1']
            audio2 = args['audio2']

            audio1_data = AudioSegment.from_file(audio1)
            audio2_data = AudioSegment.from_file(audio2)

            merged_audio = audio1_data + audio2_data

            output_file = 'merged_audio.wav'
            merged_audio.export(output_file, format='wav')

            # return send_file(output_file, mimetype='audio/wav', as_attachment=True)
            return {'merged_audio': output_file}
        except Exception as e:
            return {"message": str(e)}, 500

    #     # try:
        # args = upload_parser.parse_args()
        # uploaded_files = args['files']

        # if len(uploaded_files) < 2:
        #     return {"message": "At least two audio files are required for merging."}, 400

        # merged_audio = AudioSegment.silent(duration=0)

        # for file in uploaded_files:
        #     audio_data = AudioSegment.from_file(file, format=file.filename.split('.')[-1])
        #     merged_audio += audio_data

        # output_file = 'merged_audio.wav'
        # merged_audio.export(output_file, format="wav")


        # return send_file(output_file, mimetype="audio/wav", as_attachment=True)
        # except Exception as e:
        #     return {"message": str(e)}, 500
    def get(self, filename):
        try:
            output_buffer = io.BytesIO()
            audio = AudioSegment.from_file(filename, format=filename.split('.')[-1])
            audio.export(output_buffer, format="wav")
            output_buffer.seek(0)

            response = send_file(output_buffer, mimetype="audio/wav")
            response.headers['Content-Disposition'] = f'attachment; filename={filename}.wav'

            return response
        except Exception as e:
            return {"message": str(e)}, 500

api.add_resource(MergeAudio, '/merge/<string:filename>')
if __name__ == '__main__':
    app.run(debug=True)
