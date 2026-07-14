import os, logging, csv
import numpy as np

logger = logging.getLogger(__name__)

# --- Helper Function: Load Templates ---
def load_event_templates(file_path):
    """
    Loads event templates from a CSV file.
    Expected Format:
        Event_Type,EAE_Template
        Life:Be-Born,somebody was born in somewhere.
    """
    templates = {}
    
    if not os.path.exists(file_path):
        logger.warning(f"Event template file not found at {file_path}. Using empty templates.")
        return templates

    try:
        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            # simple CSV reader handles quotes/commas automatically
            reader = csv.DictReader(csvfile)
            
            # Check if headers match expectations
            if 'Event_Type' not in reader.fieldnames or 'EAE_Template' not in reader.fieldnames:
                logger.warning(f"CSV headers mismatch in {file_path}. Expected 'Event_Type,EAE_Template'. Found: {reader.fieldnames}")
                return templates

            for row in reader:
                evt_type = row['Event_Type'].strip()
                template = row['EAE_Template'].strip()
                
                if evt_type and template:
                    templates[evt_type] = template
                    
        logger.info(f"Successfully loaded {len(templates)} event templates from {file_path}")
        return templates

    except Exception as e:
        logger.error(f"Failed to load event templates: {e}")
        return {}


def get_span_idx(pieces, token_start_idxs, span, tokenizer, trigger_span=None):
    """
    This function is how we map the generated prediction back to span prediction.
    Detailed Explanation:
        We will first split our prediction and use tokenizer to tokenize our predicted "span" into pieces. Then, we will find whether we can find a continuous span in the original "pieces" can match tokenized "span". 
    If it is an argument/relation extraction task, we will return the one which is closest to the trigger_span.
    """
    words = []
    for s in span.split(' '):
        words.extend(tokenizer.encode(s, add_special_tokens=False))
    
    candidates = []
    for i in range(len(pieces)):
        j = 0
        k = 0
        while j < len(words) and i+k < len(pieces):
            if pieces[i+k] == words[j]:
                j += 1
                k += 1
            elif tokenizer.decode(words[j]) == "":
                j += 1
            elif tokenizer.decode(pieces[i+k]) == "":
                k += 1
            else:
                break
        if j == len(words):
            candidates.append((i, i+k))
            
    candidates = [(token_start_idxs.index(c1), token_start_idxs.index(c2)) for c1, c2 in candidates if c1 in token_start_idxs and c2 in token_start_idxs]
    if len(candidates) < 1:
        return -1, -1
    else:
        if trigger_span is None:
            return candidates[0]
        else:
            return sorted(candidates, key=lambda x: np.abs(trigger_span[0]-x[0]))[0]
        

def get_token_info(tokens, tokenizer):
    """
    Maps original whitespace-separated tokens to the LLM's subword pieces.
    
    Args:
        tokens (list of str): The original words in the document.
        tokenizer: The HuggingFace tokenizer instance.
        
    Returns:
        dict: A mapping of original token indices to their subword piece indices.
    """
    piece_idxs = []
    token_start_idxs = []
    current_idx = 0
    
    for token in tokens:
        # Get the subword pieces for this specific word
        pieces = tokenizer.tokenize(token)
        
        # Fallback for weird hidden characters that tokenizer might drop
        if not pieces: 
            pieces = [tokenizer.unk_token if tokenizer.unk_token else "<unk>"]
            
        token_start_idxs.append(current_idx)
        
        # Record the span of piece indices this word occupies
        piece_span = list(range(current_idx, current_idx + len(pieces)))
        piece_idxs.append(piece_span)
        
        current_idx += len(pieces)
        
    return {
        "piece_idxs": piece_idxs,
        "token_start_idxs": token_start_idxs
    }