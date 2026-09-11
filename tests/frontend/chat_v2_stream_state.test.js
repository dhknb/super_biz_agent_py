const assert = require('node:assert/strict');

const {
  createChatV2StreamState,
  applyChatV2StreamEvent,
  getChatV2StreamPanelPayload,
} = require('../../static/chat_v2_stream_state');

const state = createChatV2StreamState();

applyChatV2StreamEvent(state, { type: 'sub_queries', data: ['origin', 123] });
applyChatV2StreamEvent(state, { type: 'retrieved', data: { node: 'retrieve_each', count: 2 } });
applyChatV2StreamEvent(state, { type: 'retrieved', data: { node: 'retrieve_each', count: 3 } });
applyChatV2StreamEvent(state, {
  type: 'used_documents',
  data: [{ content: 'doc', metadata: { id: '1' } }],
});
applyChatV2StreamEvent(state, { type: 'answer', data: 'final answer' });
applyChatV2StreamEvent(state, {
  type: 'validation',
  data: { blocked: false, groundedness_score: 0.88 },
});
applyChatV2StreamEvent(state, { type: 'done' });

assert.equal(state.answer, 'final answer');
assert.equal(state.complete, true);
assert.deepEqual(getChatV2StreamPanelPayload(state), {
  answer: 'final answer',
  sub_queries: ['origin', '123'],
  retrieved_count: 5,
  used_documents: [{ content: 'doc', metadata: { id: '1' } }],
  validation: { blocked: false, groundedness_score: 0.88 },
});

console.log('chat_v2_stream_state tests passed');
