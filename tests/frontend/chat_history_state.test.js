const assert = require('node:assert/strict');

const {
  normalizeHistoryMessage,
  getChatPersistenceAction,
} = require('../../static/chat_history_state');

assert.deepEqual(
  normalizeHistoryMessage({ role: 'assistant', content: '### Title', timestamp: 't1' }),
  { type: 'assistant', content: '### Title', timestamp: 't1' }
);

assert.deepEqual(
  normalizeHistoryMessage({
    role: 'assistant',
    content: 'answer',
    message_metadata: {
      high_precision: {
        sub_queries: ['q1'],
        retrieved_count: 1,
        used_documents: [],
        validation: { blocked: false },
      },
    },
  }),
  {
    type: 'assistant',
    content: 'answer',
    timestamp: undefined,
    high_precision: {
      sub_queries: ['q1'],
      retrieved_count: 1,
      used_documents: [],
      validation: { blocked: false },
    },
  }
);

assert.deepEqual(
  normalizeHistoryMessage({ type: 'bot', content: '**legacy**' }),
  { type: 'assistant', content: '**legacy**', timestamp: undefined }
);

assert.deepEqual(
  normalizeHistoryMessage({ role: 'user', content: 'hello' }),
  { type: 'user', content: 'hello', timestamp: undefined }
);

assert.deepEqual(
  normalizeHistoryMessage({
    role: 'assistant',
    content: 'answer',
    high_precision: {
      sub_queries: ['q1'],
      retrieved_count: 2,
      used_documents: [],
      validation: { blocked: true },
    },
  }),
  {
    type: 'assistant',
    content: 'answer',
    timestamp: undefined,
    high_precision: {
      sub_queries: ['q1'],
      retrieved_count: 2,
      used_documents: [],
      validation: { blocked: true },
    },
  }
);

assert.equal(getChatPersistenceAction(false, [{ type: 'user', content: 'q' }]), 'save');
assert.equal(getChatPersistenceAction(true, [{ type: 'user', content: 'q' }]), 'update');
assert.equal(getChatPersistenceAction(false, []), 'none');

console.log('chat_history_state tests passed');
