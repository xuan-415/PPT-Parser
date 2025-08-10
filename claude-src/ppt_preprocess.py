import json
import requests
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PPTPreprocessor:
    def __init__(self, model_endpoint: str, model_headers: Dict[str, str] = None):
        """
        初始化PPT前處理器
        
        Args:
            model_endpoint: 本地LLM模型的API端點
            model_headers: API請求的headers
        """
        self.model_endpoint = model_endpoint
        self.model_headers = model_headers or {"Content-Type": "application/json"}
        
        # 連續性判斷的prompt模板
        self.continuity_prompt = """
你是一個專門分析PPT頁面連續性的助手。請分析以下兩個頁面的內容，判斷它們是否在邏輯上連續。

判斷標準：
1. 內容主題是否相關
2. 是否存在邏輯上的承接關係
3. 是否有明顯的段落或章節連接
4. 第二頁是否是第一頁內容的延續或深入探討

頁面1內容：
{page1_content}

頁面2內容：
{page2_content}

請只回答"連續"或"不連續"，不需要其他解釋。
"""

        # 內容改寫的prompt模板
        self.rewrite_prompt = """
你是一個專門將PPT內容轉換為適合LLM訓練的助手。請將以下PPT內容改寫為更適合語言模型學習的格式。

改寫要求：
1. 保持原有的核心信息和知識點
2. 將條列式內容轉換為連貫的段落
3. 補充必要的連接詞和過渡句
4. 確保邏輯清晰，語言流暢
5. 適當擴展簡略的表達

原始PPT內容：
{content}

請提供改寫後的內容：
"""

    def call_local_llm(self, prompt: str, max_retries: int = 3) -> str:
        """
        調用本地LLM模型
        
        Args:
            prompt: 輸入的prompt
            max_retries: 最大重試次數
            
        Returns:
            模型的回應文本
        """
        payload = {
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.model_endpoint,
                    headers=self.model_headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                # 根據你的模型API格式調整這部分
                if 'choices' in result:
                    return result['choices'][0]['text'].strip()
                elif 'response' in result:
                    return result['response'].strip()
                else:
                    return str(result).strip()
                    
            except Exception as e:
                logger.warning(f"API調用失敗 (嘗試 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"API調用失敗，已重試{max_retries}次")
        
    def extract_content_from_page(self, page_data: Dict[str, Any]) -> str:
        """
        從頁面JSON數據中提取文本內容
        
        Args:
            page_data: 頁面的JSON數據 (格式: {"file_name": "...", "page": 1, "content": "..."})
            
        Returns:
            提取的文本內容
        """
        # 直接返回content字段的內容
        return page_data.get('content', '')

    def check_continuity(self, page1: Dict[str, Any], page2: Dict[str, Any]) -> bool:
        """
        檢查兩個頁面是否連續
        
        Args:
            page1: 第一個頁面的數據
            page2: 第二個頁面的數據
            
        Returns:
            True如果頁面連續，False如果不連續
        """
        content1 = self.extract_content_from_page(page1)
        content2 = self.extract_content_from_page(page2)
        
        prompt = self.continuity_prompt.format(
            page1_content=content1,
            page2_content=content2
        )
        
        try:
            response = self.call_local_llm(prompt)
            result = response.lower().strip()
            
            if "連續" in result and "不連續" not in result:
                return True
            elif "不連續" in result:
                return False
            else:
                logger.warning(f"模型回應不明確: {response}")
                return False
                
        except Exception as e:
            logger.error(f"連續性判斷失敗: {e}")
            return False

    def merge_pages(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        合併多個頁面的內容
        
        Args:
            pages: 要合併的頁面列表 (格式: [{"file_name": "...", "page": 1, "content": "..."}, ...])
            
        Returns:
            合併後的頁面數據
        """
        if not pages:
            return {}
        
        merged = {
            "file_name": pages[0].get("file_name", ""),
            "page_numbers": [page.get("page", i+1) for i, page in enumerate(pages)],
            "page_range": f"{pages[0].get('page', 1)}-{pages[-1].get('page', len(pages))}",
            "merged_content": True,
            "original_page_count": len(pages)
        }
        
        # 合併所有內容
        all_content = []
        for page in pages:
            content = page.get('content', '')
            if content.strip():  # 只添加非空內容
                all_content.append(content.strip())
        
        merged["content"] = "\n\n".join(all_content)
        
        return merged

    def check_continuity_with_group(self, group: List[Dict[str, Any]], new_page: Dict[str, Any]) -> bool:
        """
        檢查新頁面是否與整個組連續
        
        Args:
            group: 當前的頁面組
            new_page: 要檢查的新頁面
            
        Returns:
            True如果新頁面與組連續，False如果不連續
        """
        # 方法1: 與組中的最後一頁比較
        last_page_continuity = self.check_continuity(group[-1], new_page)
        
        # 方法2: 與整個組的合併內容比較
        merged_group = self.merge_pages(group)
        group_continuity = self.check_continuity(merged_group, new_page)
        
        # 方法3: 與組中的第一頁比較 (檢查是否為同一主題)
        first_page_continuity = self.check_continuity(group[0], new_page)
        
        logger.info(f"連續性檢查結果 - 最後一頁: {last_page_continuity}, 整組: {group_continuity}, 第一頁: {first_page_continuity}")
        
        # 綜合判斷邏輯 - 可以根據需求調整
        return last_page_continuity or group_continuity or first_page_continuity

    def find_best_group_for_page(self, groups: List[List[Dict[str, Any]]], page: Dict[str, Any]) -> Tuple[int, float]:
        """
        為頁面找到最佳的組
        
        Args:
            groups: 現有的組列表
            page: 要分配的頁面
            
        Returns:
            (最佳組的索引, 相關度分數) 如果沒有合適的組則返回(-1, 0)
        """
        best_group_idx = -1
        best_score = 0.0
        
        for i, group in enumerate(groups):
            if self.check_continuity_with_group(group, page):
                # 可以添加更複雜的評分邏輯
                score = 1.0
                if score > best_score:
                    best_score = score
                    best_group_idx = i
        
        return best_group_idx, best_score

    def find_continuous_groups_advanced(self, pages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        進階的連續頁面組尋找方法，支援跳頁和複雜關聯
        
        Args:
            pages: 所有頁面的列表
            
        Returns:
            連續頁面組的列表
        """
        if not pages:
            return []
        
        logger.info(f"開始進階分析 {len(pages)} 個頁面的連續性...")
        
        groups = [[pages[0]]]  # 從第一頁開始第一組
        unassigned_pages = []
        
        # 第一輪：順序處理
        for i in range(1, len(pages)):
            current_page = pages[i]
            current_page_num = current_page.get('page', i+1)
            
            logger.info(f"處理頁面 {current_page_num}...")
            
            # 嘗試找到最佳組
            best_group_idx, score = self.find_best_group_for_page(groups, current_page)
            
            if best_group_idx >= 0:
                groups[best_group_idx].append(current_page)
                logger.info(f"頁面 {current_page_num} 加入組 {best_group_idx + 1}")
            else:
                # 暫時放入未分配列表
                unassigned_pages.append(current_page)
                logger.info(f"頁面 {current_page_num} 暫時未分配")
        
        # 第二輪：處理未分配的頁面
        logger.info(f"第二輪處理 {len(unassigned_pages)} 個未分配頁面...")
        
        for page in unassigned_pages:
            page_num = page.get('page', 'Unknown')
            
            # 再次嘗試分配到現有組
            best_group_idx, score = self.find_best_group_for_page(groups, page)
            
            if best_group_idx >= 0:
                groups[best_group_idx].append(page)
                logger.info(f"頁面 {page_num} 在第二輪加入組 {best_group_idx + 1}")
            else:
                # 創建新組
                groups.append([page])
                logger.info(f"頁面 {page_num} 創建新組 {len(groups)}")
        
        # 對每組內的頁面按頁碼排序
        for group in groups:
            group.sort(key=lambda x: x.get('page', 0))
        
        logger.info(f"完成進階連續性分析，共分為 {len(groups)} 組")
        return groups

    def find_continuous_groups(self, pages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        尋找連續的頁面組 - 提供簡單和進階兩種模式
        
        Args:
            pages: 所有頁面的列表
            
        Returns:
            連續頁面組的列表
        """
        # 可以在這裡選擇使用哪種方法
        use_advanced_method = True  # 設為False使用原始的簡單方法
        
        if use_advanced_method:
            return self.find_continuous_groups_advanced(pages)
        else:
            return self.find_continuous_groups_simple(pages)

    def find_continuous_groups_simple(self, pages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        簡單的順序連續性判斷方法（原始方法）
        """
        if not pages:
            return []
        
        groups = []
        current_group = [pages[0]]
        
        logger.info(f"開始簡單分析 {len(pages)} 個頁面的連續性...")
        
        for i in range(1, len(pages)):
            current_page_num = pages[i].get('page', i+1)
            prev_page_num = pages[i-1].get('page', i)
            
            logger.info(f"檢查頁面 {current_page_num} 與頁面 {prev_page_num} 的連續性...")
            
            # 檢查當前頁面與當前組的最後一頁是否連續
            if self.check_continuity(current_group[-1], pages[i]):
                current_group.append(pages[i])
                logger.info(f"頁面 {current_page_num} 與前一頁連續，加入當前組")
            else:
                # 不連續，開始新的組
                groups.append(current_group)
                current_group = [pages[i]]
                logger.info(f"頁面 {current_page_num} 與前一頁不連續，開始新組")
        
        # 添加最後一組
        groups.append(current_group)
        
        logger.info(f"完成簡單連續性分析，共分為 {len(groups)} 組")
        return groups

    def rewrite_content(self, content: str) -> str:
        """
        使用LLM改寫內容
        
        Args:
            content: 要改寫的原始內容
            
        Returns:
            改寫後的內容
        """
        prompt = self.rewrite_prompt.format(content=content)
        
        try:
            response = self.call_local_llm(prompt)
            return response
        except Exception as e:
            logger.error(f"內容改寫失敗: {e}")
            return content  # 如果改寫失敗，返回原始內容

    def process_ppt_data(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        處理完整的PPT數據
        
        Args:
            pages: PPT頁面數據列表
            
        Returns:
            處理後的數據列表
        """
        logger.info("開始PPT數據前處理...")
        
        # 步驟1: 尋找連續頁面組
        continuous_groups = self.find_continuous_groups(pages)
        
        # 步驟2: 合併連續頁面
        merged_pages = []
        for i, group in enumerate(continuous_groups):
            group_info = f"頁面 {group[0].get('page')}-{group[-1].get('page')} (文件: {group[0].get('file_name')})"
            logger.info(f"處理第 {i+1} 組，包含 {len(group)} 個頁面: {group_info}")
            merged_page = self.merge_pages(group)
            merged_pages.append(merged_page)
        
        # 步驟3: 改寫內容
        final_results = []
        for i, page in enumerate(merged_pages):
            logger.info(f"改寫第 {i+1}/{len(merged_pages)} 組內容...")
            
            content = page.get("content", "")
            if content:
                rewritten_content = self.rewrite_content(content)
                page["rewritten_content"] = rewritten_content
                page["original_content"] = content
            
            final_results.append(page)
        
        logger.info("PPT數據前處理完成")
        return final_results

def load_ppt_json(file_path: str) -> List[Dict[str, Any]]:
    """
    從JSON文件加載PPT數據
    
    Args:
        file_path: JSON文件路徑
        
    Returns:
        PPT頁面數據列表 (格式: [{"file_name": "...", "page": 1, "content": "..."}, ...])
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 確保數據是列表格式
    if isinstance(data, list):
        return data
    else:
        raise ValueError("JSON文件格式錯誤，期望是包含頁面數據的列表")

def save_results(results: List[Dict[str, Any]], output_path: str):
    """
    保存處理結果
    
    Args:
        results: 處理後的結果
        output_path: 輸出文件路徑
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 輸出一些統計信息
    logger.info(f"處理結果已保存到: {output_path}")
    logger.info(f"共處理了 {len(results)} 個組")
    
    for i, result in enumerate(results):
        page_range = result.get('page_range', 'Unknown')
        file_name = result.get('file_name', 'Unknown')
        logger.info(f"組 {i+1}: 文件 {file_name}, 頁面 {page_range}")

def main():
    """
    主函數
    """
    # 配置參數 - 請根據你的實際情況修改
    MODEL_ENDPOINT = "http://localhost:8000/v1/completions"  # 你的本地LLM端點
    INPUT_FILE = "ppt_data.json"  # 輸入的JSON文件
    OUTPUT_FILE = "processed_ppt_data.json"  # 輸出文件
    
    # 初始化處理器
    preprocessor = PPTPreprocessor(MODEL_ENDPOINT)
    
    try:
        # 加載數據
        logger.info(f"加載PPT數據: {INPUT_FILE}")
        pages = load_ppt_json(INPUT_FILE)
        logger.info(f"成功加載 {len(pages)} 個頁面")
        
        # 處理數據
        results = preprocessor.process_ppt_data(pages)
        
        # 保存結果
        logger.info(f"保存處理結果: {OUTPUT_FILE}")
        save_results(results, OUTPUT_FILE)
        
        logger.info("處理完成！")
        
        # 輸出統計信息
        original_pages = len(pages)
        processed_groups = len(results)
        logger.info(f"統計: 原始頁面數 {original_pages} -> 處理後組數 {processed_groups}")
        
    except Exception as e:
        logger.error(f"處理過程中發生錯誤: {e}")
        raise

if __name__ == "__main__":
    main()