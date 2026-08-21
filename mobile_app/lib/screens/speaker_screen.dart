import 'dart:async';
import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../services/websocket_service.dart';

class SpeakerScreen extends StatefulWidget {
  const SpeakerScreen({super.key});

  @override
  State<SpeakerScreen> createState() => _SpeakerScreenState();
}

class _SpeakerScreenState extends State<SpeakerScreen> with SingleTickerProviderStateMixin {
  late WebSocketService _wsService;
  bool _isListening = false;
  bool _autoPlayTTS = true;
  String _activeTranscript = '';
  final List<Map<String, String>> _chatMessages = [
    {'sender': 'Signer', 'text': 'হ্যালো! আমি ভালো আছি।', 'en': 'Hello! I am fine.'},
    {'sender': 'You', 'text': 'আপনার সাথে দেখা হয়ে খুশি হলাম।', 'en': 'Nice to meet you.'},
  ];

  final TextEditingController _replyController = TextEditingController();
  late AnimationController _micPulseController;

  @override
  void initState() {
    super.initState();
    _wsService = WebSocketService(clientType: 'speaker');
    _wsService.connect();

    _wsService.messageStream.listen((msg) {
      if (mounted && msg['type'] == 'SIGN_TRANSLATION') {
        final data = msg['data'] as Map<String, dynamic>? ?? {};
        final bn = data['label_bn'] as String? ?? '';
        final en = data['label_en'] as String? ?? '';
        setState(() {
          _chatMessages.add({'sender': 'Signer', 'text': bn, 'en': en});
        });
      }
    });

    _micPulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    );
  }

  @override
  void dispose() {
    _micPulseController.dispose();
    _wsService.dispose();
    _replyController.dispose();
    super.dispose();
  }

  void _toggleMic() {
    setState(() {
      _isListening = !_isListening;
      if (_isListening) {
        _micPulseController.repeat(reverse: true);
        _activeTranscript = 'Listening... ("ধন্যবাদ আপনাকে")';
        Timer(const Duration(seconds: 2), () {
          if (mounted && _isListening) {
            _replyController.text = 'ধন্যবাদ আপনাকে';
            _sendSpeech('ধন্যবাদ আপনাকে');
            setState(() {
              _isListening = false;
              _micPulseController.stop();
              _activeTranscript = '';
            });
          }
        });
      } else {
        _micPulseController.stop();
        _activeTranscript = '';
      }
    });
  }

  void _sendSpeech(String text) {
    if (text.trim().isEmpty) return;
    setState(() {
      _chatMessages.add({'sender': 'You', 'text': text, 'en': 'Sent Speech'});
      _replyController.clear();
    });
    _wsService.sendSpeechEvent(text);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bgDark,
      appBar: AppBar(
        title: const Text('Speaker (Hearing) Mode'),
        actions: [
          Row(
            children: [
              const Text('Auto-TTS', style: TextStyle(fontSize: 12, color: AppColors.textSecondary)),
              Switch(
                value: _autoPlayTTS,
                activeColor: AppColors.cyanAccent,
                onChanged: (val) => setState(() => _autoPlayTTS = val),
              ),
            ],
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            _buildAvatarViewport(),
            Expanded(child: _buildChatTranscriptList()),
            if (_activeTranscript.isNotEmpty) _buildListeningBanner(),
            _buildSpeechControls(),
          ],
        ),
      ),
    );
  }

  Widget _buildAvatarViewport() {
    return Container(
      height: 180,
      margin: const EdgeInsets.all(12),
      decoration: AppTheme.glassCardDecoration(),
      child: Stack(
        children: [
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Container(
                  width: 70,
                  height: 70,
                  decoration: BoxDecoration(
                    color: AppColors.surfaceDark,
                    shape: BoxShape.circle,
                    border: Border.all(color: AppColors.cyanAccent, width: 2),
                  ),
                  child: const Icon(Icons.accessibility_new_rounded, size: 40, color: AppColors.cyanGlow),
                ),
                const SizedBox(height: 8),
                const Text(
                  '3D BdSL Sign Avatar Ready',
                  style: TextStyle(fontSize: 12, color: AppColors.textSecondary, fontWeight: FontWeight.bold),
                ),
              ],
            ),
          ),
          Positioned(
            top: 8,
            right: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(color: AppColors.surfaceDark, borderRadius: BorderRadius.circular(4)),
              child: const Text('Synthesizer: Active', style: TextStyle(fontSize: 10, color: AppColors.emeraldSuccess)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildChatTranscriptList() {
    return ListView.builder(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      itemCount: _chatMessages.length,
      itemBuilder: (context, index) {
        final msg = _chatMessages[index];
        final isSigner = msg['sender'] == 'Signer';

        return Align(
          alignment: isSigner ? Alignment.centerLeft : Alignment.centerRight,
          child: Container(
            margin: const EdgeInsets.symmetric(vertical: 6),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.78),
            decoration: BoxDecoration(
              color: isSigner ? AppColors.panelDark : AppColors.cyanAccent.withOpacity(0.2),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(
                color: isSigner ? AppColors.glassBorder : AppColors.cyanAccent.withOpacity(0.5),
              ),
            ),
            child: Column(
              crossAxisAlignment: isSigner ? CrossAxisAlignment.start : CrossAxisAlignment.end,
              children: [
                Text(
                  msg['sender']!,
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    color: isSigner ? AppColors.emeraldSuccess : AppColors.cyanAccent,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  msg['text']!,
                  style: const TextStyle(fontSize: 16, color: AppColors.textPrimary),
                ),
                if (msg['en'] != null && msg['en']!.isNotEmpty)
                  Text(
                    msg['en']!,
                    style: const TextStyle(fontSize: 12, color: AppColors.textSecondary, fontStyle: FontStyle.italic),
                  ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildListeningBanner() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      color: AppColors.coralError.withOpacity(0.2),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.graphic_eq_rounded, color: AppColors.coralError, size: 20),
          const SizedBox(width: 8),
          Text(_activeTranscript, style: const TextStyle(color: AppColors.coralError, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }

  Widget _buildSpeechControls() {
    return Container(
      padding: const EdgeInsets.all(12),
      color: AppColors.panelDark,
      child: Column(
        children: [
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _replyController,
                  decoration: const InputDecoration(
                    hintText: 'Type reply to translate into BdSL...',
                    contentPadding: EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                  ),
                  onSubmitted: _sendSpeech,
                ),
              ),
              const SizedBox(width: 8),
              IconButton.filled(
                onPressed: () => _sendSpeech(_replyController.text),
                icon: const Icon(Icons.send_rounded),
                style: IconButton.styleFrom(backgroundColor: AppColors.cyanAccent, foregroundColor: AppColors.bgDark),
              ),
            ],
          ),
          const SizedBox(height: 10),
          GestureDetector(
            onTap: _toggleMic,
            child: AnimatedBuilder(
              animation: _micPulseController,
              builder: (context, child) {
                final scale = _isListening ? 1.0 + (_micPulseController.value * 0.12) : 1.0;
                return Transform.scale(
                  scale: scale,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                    decoration: BoxDecoration(
                      color: _isListening ? AppColors.coralError : AppColors.surfaceDark,
                      borderRadius: BorderRadius.circular(30),
                      border: Border.all(
                        color: _isListening ? AppColors.coralError : AppColors.cyanAccent,
                        width: 1.5,
                      ),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          _isListening ? Icons.mic : Icons.mic_none_rounded,
                          color: _isListening ? Colors.white : AppColors.cyanAccent,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          _isListening ? 'Listening (Release to Send)' : '🎤 Hold / Tap to Speak (STT)',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            color: _isListening ? Colors.white : AppColors.textPrimary,
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
