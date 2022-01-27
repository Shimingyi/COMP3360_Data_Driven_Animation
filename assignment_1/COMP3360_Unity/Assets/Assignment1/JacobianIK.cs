using System;
using System.Collections.Generic;
using UnityEngine;

public class JacobianIK : MonoBehaviour {
	// Input
	public Transform IKRoot = null;
    public Transform IKTip = null;
    public Transform IKTarget = null;
    private Transform[] bones;
	
	// Parameters
	[Range(0f, 1f)] public float Step = 0.1f;
	[Range(0f, 1f)] public float Damping = 0.1f;
	private float Differential = 0.001f;
	// Jacobian Matrix
	private int DoF;
	private int Entries;
	private int TargetNum = 1; // todo here we simply use one joint/target for demonstration
	private float[][] Jacobian;
	private float[] Gradient;

	void Start() {
        SetToZeroPose(this.transform.Find("Root"));
    }

    void Update() {
		if(IKRoot == null || IKTip == null || IKTarget == null) {
			Debug.Log("Some inspector variable has not been assigned and is still null.");
			return;
		}
		SolveByJacobian();
	}

	public void SolveByJacobian() {
  		bones = GetChain(IKRoot, IKTip);
		Matrix4x4[] posture = GetPosture();
		float[] solution = new float[3*bones.Length];
		DoF = bones.Length * 3;
		Entries = 7 * TargetNum; // 7 includes position(xyz) and rotation(xyzw)
		Jacobian = new float[Entries][];
		for(int i=0; i<Entries; i++) {
			Jacobian[i] = new float[DoF];
		}
		Gradient = new float[Entries];
		for(int i=0; i<50; i++) {
			Iterate(posture, solution);
		}
		FK(posture, solution);
	}
	
	private void FK(Matrix4x4[] posture, float[] variables) {
		for(int i=0; i<bones.Length; i++) {
			Quaternion update = Quaternion.AngleAxis(Mathf.Rad2Deg*variables[i*3+0], Vector3.forward) * Quaternion.AngleAxis(Mathf.Rad2Deg*variables[i*3+1], Vector3.right) * Quaternion.AngleAxis(Mathf.Rad2Deg*variables[i*3+2], Vector3.up);
			bones[i].localPosition = posture[i].GetPosition();
			bones[i].localRotation = posture[i].GetRotation() * update;
		}
	}

	private Matrix4x4[] GetPosture() {
		Matrix4x4[] posture = new Matrix4x4[bones.Length];
		for(int i=0; i<posture.Length; i++) {
			posture[i] = bones[i].GetLocalMatrix();
		}
		return posture;
	}

	private void Iterate(Matrix4x4[] posture, float[] variables) {
		FK(posture, variables);
		Vector3 tipPosition = IKTip.position;
		Quaternion tipRotation = IKTip.rotation;
		
		int index = 0;
		//Jacobian
		for(int j=0; j<DoF; j++) {
			variables[j] += Differential;
			FK(posture, variables);
			variables[j] -= Differential;

			index = 0;
			Vector3 deltaPosition = (IKTip.position - tipPosition) / Differential;
			Quaternion deltaRotation = Quaternion.Inverse(tipRotation) * IKTip.rotation;
	
			Jacobian[index][j] = deltaPosition.x; index += 1;
			Jacobian[index][j] = deltaPosition.y; index += 1;
			Jacobian[index][j] = deltaPosition.z; index += 1;

			Jacobian[index][j] = deltaRotation.x / Differential; index += 1;
			Jacobian[index][j] = deltaRotation.y / Differential; index += 1;
			Jacobian[index][j] = deltaRotation.z / Differential; index += 1;
			Jacobian[index][j] = (deltaRotation.w-1f) / Differential; index += 1;
		}

		//Gradient Vector
		index = 0;
		Vector3 gradientPosition = Step * (IKTarget.position - tipPosition);
		Quaternion gradientRotation = Quaternion.Inverse(tipRotation) * IKTarget.rotation;

		Gradient[index] = gradientPosition.x; index += 1;
		Gradient[index] = gradientPosition.y; index += 1;
		Gradient[index] = gradientPosition.z; index += 1;

		Gradient[index] = Step * gradientRotation.x; index += 1;
		Gradient[index] = Step * gradientRotation.y; index += 1;
		Gradient[index] = Step * gradientRotation.z; index += 1;
		Gradient[index] = Step * (gradientRotation.w-1f); index += 1;
		
		//Jacobian Damped-Least-Squares
		float[][] DLS = DampedLeastSquares();
		for(int m=0; m<DoF; m++) {
			for(int n=0; n<Entries; n++) {
				variables[m] += DLS[m][n] * Gradient[n];
			}
		}
	}


	private float[][] DampedLeastSquares() {
		// Pseudo Inverse
		float[][] transpose = MatrixUtility.MatrixCreate(DoF, Entries);
		for(int m=0; m<Entries; m++) {
			for(int n=0; n<DoF; n++) {
				transpose[n][m] = Jacobian[m][n];
			}
		}
		float[][] jTj = MatrixUtility.MatrixProduct(transpose, Jacobian);
		for(int i=0; i<DoF; i++) {
			jTj[i][i] += Damping*Damping;
		}
		float[][] dls = MatrixUtility.MatrixProduct(MatrixUtility.MatrixInverse(jTj), transpose);
		return dls;
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