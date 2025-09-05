#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dialogue_manager.intent import IntentRecognizer
from dialogue_manager.engine import EngineConfig

def test_number_entity():
    recognizer = IntentRecognizer(EngineConfig())
    result = recognizer.recognize('调节亮度到70%', {}, [])
    
    print('意图:', result['intent'])
    print('实体:')
    for e in result['entities']:
        if hasattr(e, 'entity_type'):
            print(f'  {e.entity_type}: {e.value} ({e.name})')
        else:
            print(f'  {e.get("entity_type", "unknown")}: {e.get("value", "")} ({e.get("name", "")})')

if __name__ == "__main__":
    test_number_entity()

