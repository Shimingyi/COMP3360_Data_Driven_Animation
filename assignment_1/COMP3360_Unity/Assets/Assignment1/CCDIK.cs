using System.Collections.Generic;
using UnityEngine;

public class CCDIK : MonoBehaviour {
    public Transform IKRoot = null;
    public Transform IKTip = null;
    public Transform IKTarget = null;

    void Start() {
        SetToZeroPose(this.transform.Find("Root"));
    }

    void Update() {
        if(IKRoot == null || IKTip == null || IKTarget == null) {
            Debug.Log("Some inspector variable has not been assigned and is still null.");
            return;
        }
        SolveByCCD(IKRoot, IKTip, IKTarget);
    }


    // CCDSolver
    public void SolveByCCD(Transform root, Transform tip, Transform target) {
        Transform[] chain = GetChain(root, tip);
        if(chain.Length == 0) {
            Debug.Log("Given root and tip are not valid.");
            return;
        }

        //############################################################
        //Your implementation goes here
        //############################################################
        int iterations = 50;
        bool solveRotation = true;
        bool solvePosition = true;
        for(int i=0; i<iterations; i++) {
            for(int j=0; j<chain.Length; j++) {

                //Subtask 2: Heuristic IK Weight Computation
                float weight = (float)(j+1) / (float)chain.Length;
                // float weight = Mathf.Pow((float)(j+1) / (float)chain.Length, 2f);
                // float weight = Mathf.Sqrt((float)(j+1) / (float)chain.Length);
                // float weight = 1f;

                //Subtask 1: Solve Tip Position
                /*
                Tips:
                1. Get tip/end effector position/rotation : tip.position/tip.rotation
                2. Get position/rotation of the jth joint on the hierarchical chain : chain[j].position/chain[j].rotation
                3. Get target position/rotation : target.position/target.rotation
                4. Creates a rotation(Quaternion) which rotates from fromDirection to toDirection : Quaternion.FromToRotation(Vector3 fromDirection, Vector3 toDirection);
                5. Rotate the jth joint on the chain by a Rotation R (in quaternion): chain[j].rotation = R * chain[j].rotation rotations
                */
               
				if(solvePosition) {
					chain[j].rotation = Quaternion.Slerp(
						chain[j].rotation,
						Quaternion.FromToRotation(tip.position - chain[j].position, target.position - chain[j].position) * chain[j].rotation,
						weight
					);
				}
            }
        }
        //############################################################
    }

    public static Transform[] GetChain(Transform root, Transform end) {
        if(root == null || end == null) {
            return new Transform[0];
        }
        List<Transform> chain = new List<Transform>();
        Transform joint = end;
        chain.Add(joint);
        while(joint != root) {
            joint = joint.parent;
            if(joint == null) {
                return new Transform[0];
            } else {
                chain.Add(joint);
            }
        }
        chain.Reverse();
        return chain.ToArray();
    }

    public static void SetToZeroPose(Transform characterRoot){
        if(characterRoot == null){
            return;
        }
        Transform[] transforms = characterRoot.GetComponentsInChildren<Transform>();
        for(int i=1; i<transforms.Length; i++){
            if(transforms[i].parent.name == "Root"){
                transforms[i].position = new Vector3(0f, transforms[i].position.y, 0f);
            }
            transforms[i].localRotation = Quaternion.identity;
        }
    }

}
