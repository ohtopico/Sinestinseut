from flask import Flask, request, jsonify, send_from_directory, render_template_string
import numpy as np
from scipy.io.wavfile import write as write_wav
import os

app = Flask(__name__)

# --- Constantes para Geração de Áudio ---
SAMPLE_RATE = 44100 # 44.1kHz
DURATION_PER_SLICE = 0.05 # 50 ms por fatia de pixel X. Total: 800 * 0.05 = 40 segundos.

# Frequências base para o mapeamento de notas (A4 = 440 Hz)
A4 = 440.0
C4 = A4 * (2**(-9/12)) # ~261.63 Hz

# --- Funções de Mapeamento e Síntese ---

def hue_to_frequency(hue):
    """
    Mapeia o Matiz (Hue, 0-360) para frequência musical (C4 a C5).
    Hue 0/360 (vermelho) = C4; Hue 30 (laranja) = C#4; ... Hue 330 (magenta) = B4.
    """
    # Escala de 12 semitons
    semitones = (hue / 360.0) * 12 
    
    # Mapeamento exponencial (frequência)
    frequency = C4 * (2**(semitones/12))
    return frequency

def generate_waveform(frequency, duration, wave_type):
    """Gera uma forma de onda (seno, quadrado ou triangular)."""
    num_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    if wave_type == 'sine':
        # Onda Senoidal
        return np.sin(2 * np.pi * frequency * t)
    elif wave_type == 'square':
        # Onda Quadrada (Rico em harmônicos ímpares)
        return np.sign(np.sin(2 * np.pi * frequency * t))
    elif wave_type == 'triangle':
        # Onda Triangular
        return 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1
    else:
        return np.zeros_like(t)

def generate_sound_from_data(drawing_data):
    """Processa os dados HSV e gera o sinal de áudio."""
    audio_signal = np.array([], dtype=np.float32)

    for h, s, v in drawing_data:
        # 1. Frequência (Nota) - Mapeamento do Hue (0-360)
        freq = hue_to_frequency(h)
        
        # 2. Timbre (Forma de Onda) - Mapeamento da Saturação (0-1)
        if s < 0.33:
            wave_type = 'sine'
        elif s < 0.67:
            wave_type = 'triangle'
        else:
            wave_type = 'square'

        # 3. Volume (Amplitude) - Mapeamento do Brilho (Value, 0-1)
        amplitude = v * 0.5 # Multiplicador de 0.5 para evitar clipping
        
        # Gera o segmento e aplica o volume
        segment = generate_waveform(freq, DURATION_PER_SLICE, wave_type)
        audio_signal = np.concatenate([audio_signal, segment * amplitude])

    # Normaliza o sinal final
    if len(audio_signal) > 0:
        max_amplitude = np.max(np.abs(audio_signal))
        if max_amplitude > 0:
            audio_signal /= max_amplitude
    
    # Converte para formato de 16 bits (padrão)
    int_audio = np.int16(audio_signal * 32767)
    
    # Salva o arquivo .wav na pasta "static" para ser acessível pelo navegador
    output_path = os.path.join(app.root_path, 'static', 'output.wav')
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # Cria a pasta 'static' se não existir
    write_wav(output_path, SAMPLE_RATE, int_audio)
    
    return 'output.wav'

# --- Rotas do Flask ---

@app.route('/')
def index():
    """Rota para servir o arquivo HTML principal."""
    # Como o index.html está no mesmo diretório, ele é servido diretamente
    return send_from_directory(app.root_path, 'index.html')

@app.route('/styles.css')
def serve_css():
    """Rota para servir o arquivo CSS."""
    return send_from_directory(app.root_path, 'styles.css')

@app.route('/generate-sound', methods=['POST'])
def handle_drawing_data():
    """Rota para receber os dados do desenho, gerar o som e retornar o sucesso."""
    try:
        data = request.json
        drawing_data = data.get('drawing_data')
        
        if not drawing_data:
            return jsonify({'error': 'Nenhum dado de desenho recebido.'}), 400

        # Chama a função de síntese
        filename = generate_sound_from_data(drawing_data)
        
        # Retorna o nome do arquivo gerado
        return jsonify({'message': 'Áudio gerado com sucesso.', 'filename': filename}), 200

    except Exception as e:
        print(f"Erro no processamento: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    """Rota para servir o arquivo .wav gerado."""
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

if __name__ == '__main__':
    # Cria a pasta 'static' se ela não existir
    os.makedirs(os.path.join(app.root_path, 'static'), exist_ok=True)
    print("--- Servidor do Sintetizador Colorido ---")
    print(f"Acesse: http://127.0.0.1:5000/")
    # Debug=True para recarregar automaticamente durante o desenvolvimento
    app.run(debug=True)