package again250224;

import java.util.ArrayList;
import java.util.List;

/**
 * 设计浏览器历史记录
 */
public class BrowserHistory {

    private final List<String> history = new ArrayList<>();
    private int cur;

    public BrowserHistory(String homepage) {
        history.add(homepage);
    }

    public void visit(String url) {
        cur++;
        history.subList(cur, history.size()).clear();
        history.add(url);
    }

    public String back(int steps) {
        cur = Math.max(cur - steps, 0);
        return history.get(cur);
    }

    public String forward(int steps) {
        cur = Math.min(cur + steps, history.size() - 1);
        return history.get(cur);
    }

}
