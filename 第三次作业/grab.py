from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os

# ========== 1. 配置 Edge 浏览器 ==========
edge_options = Options()
edge_options.add_argument("--start-maximized")
edge_options.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Edge(options=edge_options)
wait = WebDriverWait(driver, 60)

# ========== 2. 打开 ESI 页面 ==========
url = "https://esi.clarivate.com/IndicatorsAction.action?app=esi&Init=Yes"
driver.get(url)
input("👉 请在浏览器中完成 CERNET 登录并进入 ESI 主页面后，按回车继续...")

# ========== 3. 创建保存目录 ==========
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "esi_results")
os.makedirs(save_dir, exist_ok=True)

# ========== 4. 定义表格抓取函数 ==========
def scrape_table(field_name):
    try:
        # 1) 等到表格主体出现（动态 id）
        tbody_sel = "tbody[id^='gridview-'][id$='-body']"
        table_body = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, tbody_sel))
        )

        # 2) 定位滚动容器（动态 id / 类名兜底）
        try:
            scroll_container = driver.find_element(By.CSS_SELECTOR, "div[id^='gridview-']")
        except:
            scroll_container = driver.find_element(By.CSS_SELECTOR, "div.x-grid-view")

        # 3) 增量滚动 & 去重累计
        seen_keys = set()
        all_rows = []

        def harvest_visible_rows():
            rows = driver.find_elements(By.CSS_SELECTOR, f"{tbody_sel} tr")
            added = 0
            for r in rows:
                # 先取稳定的行标识
                key = r.get_attribute("id") or r.get_attribute("data-recordid")
                tds = r.find_elements(By.TAG_NAME, "td")
                if not key:
                    # 兜底：用前三列内容做去重键（排名+机构+国家）
                    head = [td.text.strip() for td in tds[:3]]
                    key = "|".join(head)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                values = [td.text.strip() for td in tds]
                if values:
                    all_rows.append(values)
                    added += 1
            return added

        # 先收集首屏
        harvest_visible_rows()

        # 滚动参数
        stuck_rounds = 0
        prev_top = -1

        while True:
            # 读取当前滚动位置与最大值
            cur_top = driver.execute_script("return arguments[0].scrollTop;", scroll_container)
            max_top = driver.execute_script("return arguments[0].scrollHeight - arguments[0].clientHeight;", scroll_container)

            # 向下滚动一个视窗高度（略小于 1 屏，触发加载）
            driver.execute_script(
                "arguments[0].scrollTop = arguments[0].scrollTop + arguments[0].clientHeight*0.9;",
                scroll_container
            )
            time.sleep(0.5)

            # 采集当前屏的行
            added_now = harvest_visible_rows()

            # 进度判断
            new_top = driver.execute_script("return arguments[0].scrollTop;", scroll_container)
            if new_top == prev_top:
                stuck_rounds += 1
            else:
                stuck_rounds = 0
                prev_top = new_top

            # 到底 or 多次不再滚动，则再补采一次后退出
            if new_top >= max_top or stuck_rounds >= 3:
                time.sleep(0.5)
                harvest_visible_rows()
                break

        # 4) 保存 Excel
        safe_name = field_name.replace(" ", "_").replace("/", "_")
        save_path = os.path.join(save_dir, f"{safe_name}.xlsx")
        df = pd.DataFrame(all_rows)
        df.to_excel(save_path, index=False, header=False)
        print(f"✅ {field_name} 采集完成：共 {len(all_rows)} 行，已保存到 {save_path}")

    except Exception as e:
        print(f"❌ 抓取 {field_name} 失败: {e}")

# ========== 5. 主循环 ==========
while True:
    field = input("\n👉 手动切换学科后，在这里输入学科名称（或输入 q 退出）: ").strip()
    if field.lower() == "q":
        print("程序结束，关闭浏览器。")
        break
    scrape_table(field)

driver.quit()
