import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

enum ConnectionStatus { disconnected, connecting, connected }

class WebSocketService {
  WebSocketChannel? _channel;
  final String serverUrl;
  final String roomId;
  final String clientType; // 'signer' or 'speaker'

  final _statusController = StreamController<ConnectionStatus>.broadcast();
  final _messageController = StreamController<Map<String, dynamic>>.broadcast();

  Stream<ConnectionStatus> get statusStream => _statusController.stream;
  Stream<Map<String, dynamic>> get messageStream => _messageController.stream;

  bool _isDisposed = false;
  Timer? _reconnectTimer;

  WebSocketService({
    this.serverUrl = 'ws://127.0.0.1:8000',
    this.roomId = 'room_public_01',
    this.clientType = 'signer',
  });

  void connect() {
    if (_isDisposed) return;
    _statusController.add(ConnectionStatus.connecting);

    final cleanUrl = serverUrl.replaceAll('http://', 'ws://').replaceAll('https://', 'wss://');
    final wsUri = Uri.parse('$cleanUrl/ws/$clientType/$roomId');

    try {
      _channel = WebSocketChannel.connect(wsUri);
      _statusController.add(ConnectionStatus.connected);

      _channel!.stream.listen(
        (data) {
          try {
            final parsed = jsonDecode(data as String) as Map<String, dynamic>;
            _messageController.add(parsed);
          } catch (e) {
            debugPrint('Error parsing incoming WS message: $e');
          }
        },
        onError: (err) {
          debugPrint('WebSocket error: $err');
          _handleDisconnect();
        },
        onDone: () {
          debugPrint('WebSocket connection closed');
          _handleDisconnect();
        },
      );
    } catch (e) {
      debugPrint('WebSocket connect attempt failed: $e');
      _handleDisconnect();
    }
  }

  void _handleDisconnect() {
    _statusController.add(ConnectionStatus.disconnected);
    if (!_isDisposed) {
      _reconnectTimer?.cancel();
      _reconnectTimer = Timer(const Duration(seconds: 3), () {
        connect();
      });
    }
  }

  void sendSignEvent(Map<String, dynamic> signData) {
    if (_channel != null) {
      final payload = jsonEncode({
        'type': 'SIGN_TRANSLATION',
        'data': signData,
        'timestamp': DateTime.now().toIso8601String(),
      });
      _channel!.sink.add(payload);
    }
  }

  void sendSpeechEvent(String transcript) {
    if (_channel != null) {
      final payload = jsonEncode({
        'type': 'SPEECH_TEXT',
        'data': {'transcript': transcript},
        'timestamp': DateTime.now().toIso8601String(),
      });
      _channel!.sink.add(payload);
    }
  }

  void dispose() {
    _isDisposed = true;
    _reconnectTimer?.cancel();
    _channel?.sink.close();
    _statusController.close();
    _messageController.close();
  }
}
