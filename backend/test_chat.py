#!/usr/bin/env python3
"""
Test script for debugging chat functionality
"""
import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from app.db.session import SessionLocal
from app.services.chat_engine import ChatEngine
from app.models.conversation import Conversation, Message


async def test_chat():
    """Test the chat engine directly"""
    db = SessionLocal()

    try:
        print("=" * 60)
        print("CHAT ENGINE TEST")
        print("=" * 60)

        # Create a test conversation
        print("\n1. Creating test conversation...")
        conversation = Conversation(
            title="Test Conversation",
            document_id=None  # Test without specific document
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        print(f"   ✓ Created conversation ID: {conversation.id}")

        # Save a test user message
        print("\n2. Saving test user message...")
        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content="What is in the document?"
        )
        db.add(user_message)
        db.commit()
        print(f"   ✓ Saved message ID: {user_message.id}")

        # Test the chat engine
        print("\n3. Testing ChatEngine...")
        chat_engine = ChatEngine(db)

        print("   - Loading conversation history...")
        history = await chat_engine._load_conversation_history(conversation.id)
        print(f"   ✓ Loaded {len(history)} previous messages")

        print("   - Searching for context...")
        context = await chat_engine._search_context("test query", None, k=5)
        print(f"   ✓ Found {len(context)} context chunks")

        if context:
            print("   - Finding related media...")
            media = await chat_engine._find_related_media(context)
            print(f"   ✓ Found {len(media.get('images', []))} images, {len(media.get('tables', []))} tables")
        else:
            print("   ⚠ No context found - skipping media search")
            print("   ℹ This is normal if no documents have been uploaded")

        print("\n4. Processing full message...")
        result = await chat_engine.process_message(
            conversation_id=conversation.id,
            message="What is in the document?",
            document_id=None
        )

        print(f"   ✓ Got response in {result['processing_time']:.2f}s")
        print(f"   Answer: {result['answer'][:100]}...")
        print(f"   Sources: {len(result['sources'])} items")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        db.close()


if __name__ == "__main__":
    print("Starting chat engine tests...")
    asyncio.run(test_chat())
