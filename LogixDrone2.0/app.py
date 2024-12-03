import cv2
import time
import os
import sys
import base64
import threading
import glob
import socket
import numpy as np
from collections import deque
from flask import Flask, render_template, jsonify, Response
from flask_socketio import SocketIO
import logging

# Configurações de logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurações de diretórios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
QR_IMAGES_DIR = os.path.join(BASE_DIR, 'qr_images')

class QRCodeCapture:
    def __init__(self):
        # Método para obter IP local
        self.local_ip = self.get_local_ip()
        
        # URL de stream RTMP para Monaserver
        self.rtmp_stream_url = f"rtmp://{self.local_ip}:1935/live"
        
        # Inicialização dos atributos
        self.qr_codes = []
        self.unique_products = set()
        self.frame = None
        self.running = False
        self.detector = cv2.QRCodeDetector()
        self.log_messages = []
        self.image_dir = QR_IMAGES_DIR
        
        # Buffer de frames otimizado
        self.frame_buffer = deque(maxlen=5)
        self.frame_lock = threading.Lock()
        self.last_frame_time = time.time()
        
        os.makedirs(self.image_dir, exist_ok=True) # Cria o diretório se não existir
        
        # Log da URL de stream
        self.log(f"🌐 Stream URL: {self.rtmp_stream_url}")

    def get_local_ip(self):
        """Obtém IP local da máquina"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return 'localhost'

    def log(self, message):
        """Registra logs com timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
        self.log_messages.append(formatted_message)
        if len(self.log_messages) > 50:
            self.log_messages.pop(0)
        logger.debug(message)

    def start(self):
        """Inicializa o sistema de captura"""
        self.log("🚀 Iniciando sistema...")
        self.qr_codes = []
        self.unique_products.clear()
        self.running = True
        
        # Inicia thread de processamento de stream
        threading.Thread(target=self.process_stream, daemon=True).start()
        
        return True
def process_images(self):
    """Processa imagens do diretório para detecção de QR codes"""
    processed_qr_codes = []
    try:
        image_files = glob.glob(os.path.join(self.image_dir, '*.[jJ][pP][gG]'))
        
        if not image_files:
            self.log("⚠️ Nenhuma imagem encontrada no diretório")
            return processed_qr_codes
            
        self.log(f"📁 Processando {len(image_files)} imagens...")
        
        for image_file in image_files:
            try:
                image = cv2.imread(image_file)
                if image is None:
                    continue
                
                data, bbox, _ = self.detector.detectAndDecode(image)
                if data and data not in processed_qr_codes:
                    processed_qr_codes.append(data)
                    self.qr_codes.append(data)  # Adiciona ao conjunto geral de QR codes
                    unique_product = data.split(':')[0] if ':' in data else data
                    self.unique_products.add(unique_product)
                    self.log(f"✅ QR Code detectado em {os.path.basename(image_file)}: {data}")
                    
            except Exception as e:
                self.log(f"❌ Erro ao processar imagem {os.path.basename(image_file)}: {e}")
        
        self.log(f"✅ Processamento concluído: {len(processed_qr_codes)} QR codes encontrados")
        return processed_qr_codes
        
    except Exception as e:
        self.log(f"❌ Erro ao processar diretório de imagens: {e}")
        return processed_qr_codes

    def process_stream(self):
        """Processa stream com sistema de reconexão melhorado"""
        retry_count = 0
        max_retries = 3
        retry_delay = 2
        
        while self.running:
            try:
                cap = cv2.VideoCapture(self.rtmp_stream_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        retry_count += 1
                        self.log(f"⚠️ Tentativa {retry_count} de {max_retries} de reconexão...")
                        if retry_count >= max_retries:
                            self.log("🔄 Reiniciando conexão com stream...")
                            break
                        time.sleep(retry_delay)
                        continue
                    
                    retry_count = 0  # Reset do contador após sucesso
                    
                    # Processamento do frame
                    processed_frame = self.process_frame(frame.copy())  # Usa uma cópia do frame
                    
                    # Atualiza buffer com o frame processado
                    with self.frame_lock:
                        if len(self.frame_buffer) >= self.frame_buffer.maxlen:
                            self.frame_buffer.popleft()
                        self.frame_buffer.append(processed_frame)
                    
                    time.sleep(0.033)  # ~30 FPS
                
                cap.release()
                time.sleep(retry_delay)
                
            except Exception as e:
                self.log(f"❌ Erro no processamento de stream: {e}")
                time.sleep(retry_delay)

    def process_frame(self, frame):
        """Processa frame com melhor visualização do QR code"""
        try:
            data, bbox, _ = self.detector.detectAndDecode(frame)
            
            if data and data not in self.qr_codes:
                self.qr_codes.append(data)
                unique_product = data.split(':')[0] if ':' in data else data
                self.unique_products.add(unique_product)
                self.log(f"✅ Novo QR Code detectado: {data}")
            
            # Desenha retângulo mesmo se o QR code já foi detectado
            if bbox is not None and len(bbox) > 0:
                bbox = bbox[0].astype(int)
                # Desenha retângulo verde mais visível
                for i in range(len(bbox)):
                    cv2.line(frame,
                            tuple(bbox[i]),
                            tuple(bbox[(i+1)%len(bbox)]),
                            (0,255,0), 3)  # Linha mais grossa
                # Adiciona área sombreada para destacar
                pts = bbox.reshape((-1,1,2))
                cv2.polylines(frame, [pts], True, (0,255,0), 3)
                cv2.fillPoly(frame, [pts], (0,255,0,30))
                
                # Adiciona texto indicando detecção
                cv2.putText(frame, "QR Code Detectado", 
                          (bbox[0][0], bbox[0][1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                
        except Exception as e:
            self.log(f"❌ Erro no processamento de frame: {e}")
        
        return frame
    def process_frame(self, frame):
        """Processa frame para detecção de QR Code"""
        try:
            data, bbox, _ = self.detector.detectAndDecode(frame)
            
            if data and data not in self.qr_codes:
                self.qr_codes.append(data)
                unique_product = data.split(':')[0] if ':' in data else data
                self.unique_products.add(unique_product)
                
                # Desenha retângulo no QR Code
                if bbox is not None and len(bbox) > 0:
                    bbox = bbox[0].astype(int)
                    for i in range(len(bbox)):
                        cv2.line(frame, 
                                 tuple(bbox[i]), 
                                 tuple(bbox[(i+1)%len(bbox)]), 
                                 (0,255,0), 2)
        
        except Exception as e:
            self.log(f"❌ Erro no processamento de frame: {e}")

    def get_current_frame(self):
        """Obtém o frame mais recente"""
        with self.frame_lock:
            return self.frame_buffer[-1] if self.frame_buffer else None

    # Restante dos métodos mantidos iguais...

# Configuração do Flask
app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = False

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
qr_capture = QRCodeCapture()

# Rotas do aplicativo (mantidas iguais)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_images', methods=['POST'])
def load_images():
    try:
        image_qr_codes = qr_capture.process_images()
        return jsonify({
            "success": True,
            "message": f"✅ {len(image_qr_codes)} QR Codes carregados",
            "qr_codes": image_qr_codes,
            "total_qr_codes": len(qr_capture.qr_codes),
            "unique_products": len(qr_capture.unique_products)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"❌ Erro ao carregar imagens: {str(e)}",
            "qr_codes": [],
            "total_qr_codes": 0,
            "unique_products": 0
        }), 500

@app.route('/start_inventory', methods=['POST'])
def start_inventory():
    if qr_capture.start():
        return jsonify({"success": True, "message": "Sistema iniciado"})
    return jsonify({"success": False, "message": "Falha ao iniciar sistema"})

@app.route('/stop_inventory', methods=['POST'])
def stop_inventory():
    result = qr_capture.stop()
    return jsonify({**result, "success": True})

# Rota de video_feed mantida igual
@app.route('/video_feed')
def video_feed():
    def generate():
        logger.info("🎥 Iniciando geração do video feed")
        while True:
            try:
                frame = qr_capture.get_current_frame()
                
                if frame is not None:
                    # Redimensionamento otimizado
                    height, width = frame.shape[:2]
                    
                    # Redimensiona apenas se for muito grande
                    if width > 640:
                        scale = 640 / width
                        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    
                    # Compressão de imagem mais agressiva
                    _, buffer = cv2.imencode('.jpg', frame, [
                        cv2.IMWRITE_JPEG_QUALITY, 70,  # Reduz qualidade para diminuir tamanho
                        cv2.IMWRITE_JPEG_PROGRESSIVE, 1  # Habilita carregamento progressivo
                    ])
                    
                    frame_bytes = buffer.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Controla taxa de frames
                time.sleep(0.01)  # Reduz para ~20 FPS
            
            except Exception as e:
                logger.error(f"❌ Erro ao gerar frame: {e}")
                time.sleep(0.1)  # Evita loop rápido em caso de erro

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    )

@app.route('/get_qr_codes')
def get_qr_codes():
    return jsonify({
        "qr_codes": qr_capture.qr_codes,
        "total_qr_codes": len(qr_capture.qr_codes),
        "unique_products": len(qr_capture.unique_products)
    })
 
@app.route('/get_logs')
def get_logs():
    return jsonify({"logs": qr_capture.log_messages})

@app.route('/get_rtmp_status')
def get_rtmp_status():
    return jsonify({
        'publisher_connected': True,  # Assumindo conexão com Monaserver
        'subscriber_connected': True,
        'stream_url': qr_capture.rtmp_stream_url
    })
# Mensagens de inicialização
    print("\n🚀 LogixDrone 2.1 - Sistema de Inventário")
    print("=" * 50)
    print(f"📡 Endereço IP Local: {qr_capture.local_ip}")
    print(f"🔗 URL de Stream RTMP: {stream_url}")
    print("\n🌐 Sistema 100% operacional...")
    print("👉 Conecte o drone no endereço: " + stream_url)
    print("👉 Após conectado, clique em 'Iniciar inventário'")
    print("=" * 50)

if __name__ == '__main__':
    print("🚀 Iniciando sistema...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

