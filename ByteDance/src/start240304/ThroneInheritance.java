package start240304;

import java.util.*;

/**
 * 王位继承顺序
 */
public class ThroneInheritance {

    Map<String, List<String>> edges;
    Set<String> dead;
    String king;

    public ThroneInheritance(String kingName) {
        edges = new HashMap<>();
        dead = new HashSet<>();
        king = kingName;
    }

    public void birth(String parentName, String childName) {
        List<String> children = edges.getOrDefault(parentName, new ArrayList<>());
        children.add(childName);
        edges.put(parentName, children);
    }

    public void death(String name) {
        dead.add(name);
    }

    public List<String> getInheritanceOrder() {
        List<String> ans = new ArrayList<>();
        preorder(king, ans);
        return ans;
    }

    private void preorder(String king, List<String> ans) {
        if (!dead.contains(king)) {
            ans.add(king);
        }
        List<String> children = edges.getOrDefault(king, new ArrayList<>());
        for (String child : children) {
            preorder(child, ans);
        }

    }
}
